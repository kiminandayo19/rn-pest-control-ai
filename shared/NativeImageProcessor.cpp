#include "NativeImageProcessor.h"

AAssetManager *facebook::react::g_assetManager = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_pest_1control_1mobile_NativeImageProcessor_initAssetManager(
    JNIEnv *env, jclass, jobject assetMgr) {
  /*
   * Initialize access to assets dirs
   * @return AAssetDir
   */
  facebook::react::g_assetManager = AAssetManager_fromJava(env, assetMgr);
}

namespace facebook::react {
NativeImageProcessor::NativeImageProcessor(
    std::shared_ptr<CallInvoker> jsInvoker)
    : NativeImageProcessorCxxSpec(std::move(jsInvoker)) {}

void NativeImageProcessor::loadModel(jsi::Runtime &rt) {
  try {
    const char *model_path = "best_quant.onnx";
    AAsset *asset =
        AAssetManager_open(g_assetManager, model_path, AASSET_MODE_BUFFER);

    if (!asset) {
      AAsset_close(asset);
      logToJSConsole(rt, "Failed to load assets: " + std::string(model_path));
      return;
    }

    const void *model_buffer = AAsset_getBuffer(asset);
    size_t model_size = AAsset_getLength(asset);

    if (!model_buffer) {
      AAsset_close(asset);
      logToJSConsole(rt, "Failed to retieve model pointer");
      return;
    }

    if (!ort_env) {
      ort_env =
          std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    ort_session = std::make_unique<Ort::Session>(*ort_env, model_buffer,
                                                 model_size, session_options);

    AAsset_close(asset);
    logToJSConsole(rt, "ONNX model loaded successfully");
  } catch (const Ort::Exception &e) {
    logToJSConsole(rt, "Error Loading Model: " + std::string(e.what()));
  }
}

jsi::Array NativeImageProcessor::predict(jsi::Runtime &rt,
                                         const std::string &path,
                                         const std::array<int, 4> &dims) {
  try {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Load image -> Normalize
    cv::Mat image = load_image(rt, path, dims);

    // Convert to [B, C, H, W] as onnx expect input in that form
    // Split from HCW -> H, C, W
    std::vector<cv::Mat> channels(3);
    // The code below will group each channel as H*W
    // R-Channel => [H, W]
    // G-Channel => [H, W]
    // B-Channel => [H, W]
    cv::split(image, channels);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c) {
      input_tensor_values.insert(input_tensor_values.end(),
                                 (float *)channels[c].data,
                                 (float *)channels[c].data + dims[2] * dims[3]);
    }

    int mid_y = dims[2] / 2;
    int mid_x = dims[3] / 2;
    [[maybe_unused]] float r_val = channels[0].at<float>(mid_y, mid_x);
    [[maybe_unused]] float g_val = channels[1].at<float>(mid_y, mid_x);
    [[maybe_unused]] float b_val = channels[2].at<float>(mid_y, mid_x);

    std::array<int64_t, 4> dims_64t;
    for (int i = 0; i < dims.size(); ++i) {
      dims_64t[i] = static_cast<int64_t>(dims[i]);
    }

    // Alloc input tensor
    // Run using cpu
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        dims_64t.data(), dims_64t.size());

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names = ort_session->GetInputNames();
    std::vector<const char *> input_name_ptrs;
    for (auto &name : input_names)
      input_name_ptrs.push_back(name.c_str());

    std::vector<std::string> output_names = ort_session->GetOutputNames();
    std::vector<const char *> output_name_ptrs;
    for (auto &name : output_names)
      output_name_ptrs.push_back(name.c_str());

    auto output_tensors =
        ort_session->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(),
                         &input_tensor, 1, output_name_ptrs.data(), 1);

    // Process output tensor and extract bounding boxes
    auto &output_tensor = output_tensors[0];
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    float *output_data = output_tensor.GetTensorMutableData<float>();

    auto processed_data = processDetections(
        rt, output_data, output_shape, static_cast<long>(dims_64t[2]),
        static_cast<long>(dims_64t[3]), 0.25f, 0.4f);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    logToJSConsole(rt, std::format("Prediction completed in: {} ms.",
                                   std::to_string(duration.count())));

    return processed_data;
  } catch (const std::exception &e) {
    logToJSConsole(rt, "Failed " + std::string(e.what()));
    return jsi::Array(rt, 0);
  }
}

void NativeImageProcessor::registerVisionCameraExtension(jsi::Runtime &rt) {
  // Add [this] to capture the class instance
  // auto myPlugin = [this](jsi::Runtime &rt, const jsi::Value &thisArgs, const
  // jsi::Value *args, size_t count) -> jsi::Value
  // {
  //   try
  //   {
  //     auto _frame = get_frame(rt, args, count);
  //
  //     [[maybe_unused]] auto width = _frame->get(rt,
  //     jsi::PropNameID::forUtf8(rt, "width")).asNumber();
  //     [[maybe_unused]] auto height = _frame->get(rt,
  //     jsi::PropNameID::forUtf8(rt, "height")).asNumber(); auto pixel_format =
  //     _frame->get(rt, jsi::PropNameID::forUtf8(rt,
  //     "pixelFormat")).asString(rt).utf8(rt); auto get_native_buffer_fn =
  //     _frame->get(rt, jsi::PropNameID::forUtf8(rt,
  //     "getNativeBuffer")).asObject(rt).asFunction(rt);
  //
  //     // Get buffer native store it as pointer in uint8
  //     auto buffer_obj = get_native_buffer_fn.call(rt,
  //     jsi::Object::createFromHostObject(rt, _frame)).asObject(rt); auto
  //     pointer_bigInt = buffer_obj.getProperty(rt, "pointer").asBigInt(rt);
  //     uintptr_t pointer = pointer_bigInt.getUint64(rt);
  //     AHardwareBuffer *hardware_buffer = reinterpret_cast<AHardwareBuffer
  //     *>(pointer);
  //
  //     // Get buffer description
  //     AHardwareBuffer_Desc buffer_description;
  //     AHardwareBuffer_describe(hardware_buffer, &buffer_description);
  //
  //     // Lock buffer for reading
  //     void *buffer;
  //     int result = AHardwareBuffer_lock(hardware_buffer,
  //     AHARDWAREBUFFER_USAGE_CPU_READ_MASK, -1, nullptr, &buffer); if (result
  //     != 0)
  //     {
  //       throw jsi::JSError(rt, "Failed to lock hardware buffer for
  //       reading!.");
  //     }
  //
  //     cv::Mat image;
  //     if (buffer_description.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM)
  //     {
  //       cv::Mat rgba(buffer_description.height, buffer_description.width,
  //       CV_8UC4);
  //
  //       for (uint32_t y = 0; y < buffer_description.height; y++)
  //       {
  //         std::memcpy(
  //             rgba.data + (y * buffer_description.width * 4),
  //             (uint8_t *)buffer + (y * buffer_description.stride * 4),
  //             buffer_description.width * 4);
  //       }
  //       cv::cvtColor(rgba, image, cv::COLOR_RGBA2RGB);
  //     }
  //     else if (buffer_description.format ==
  //     AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420)
  //     {
  //       cv::Mat gray(buffer_description.height, buffer_description.width,
  //       CV_8UC1);
  //
  //       for (uint32_t y = 0; y < buffer_description.height; y++)
  //       {
  //         std::memcpy(
  //             gray.data + (y * buffer_description.width),
  //             (uint8_t *)buffer + (y * buffer_description.stride),
  //             buffer_description.width);
  //       }
  //       cv::cvtColor(gray, image, cv::COLOR_GRAY2RGB);
  //     }
  //     else
  //     {
  //       cv::Mat gray(buffer_description.height, buffer_description.width,
  //       CV_8UC1); for (uint32_t y = 0; y < buffer_description.height; y++)
  //       {
  //         std::memcpy(
  //             gray.data + (y * buffer_description.width),
  //             (uint8_t *)buffer + (y * buffer_description.stride),
  //             buffer_description.width);
  //       }
  //       cv::cvtColor(gray, image, cv::COLOR_GRAY2RGB);
  //     }
  //
  //     // Unlock buffer
  //     AHardwareBuffer_unlock(hardware_buffer, nullptr);
  //     auto deleteFn = buffer_obj.getProperty(rt,
  //     "delete").asObject(rt).asFunction(rt); deleteFn.call(rt);
  //
  //     logToJSConsole(rt, matToString(image));
  //
  //     int64_t width_resize = 640;
  //     int64_t height_resize = 640;
  //
  //     // Resize to (w, h)
  //     cv::resize(image, image, cv::Size(width_resize, height_resize));
  //     image.convertTo(image, CV_32FC3, 1.0 / 255.0); // Normalize image
  //
  //     // Start inference time calculation
  //     auto start_time = std::chrono::high_resolution_clock::now();
  //
  //     // Create input tensor
  //     std::vector<int64_t>
  //         input_shape = {1, 3, height_resize, width_resize};
  //     size_t input_tensor_size = 1 * 3 * height_resize * width_resize;
  //
  //     std::vector<cv::Mat> channels(3);
  //     cv::split(image, channels);
  //
  //     std::vector<float> input_tensor_values;
  //     input_tensor_values.reserve(input_tensor_size);
  //
  //     for (int c = 0; c < 3; ++c)
  //     {
  //       // Insert all data from channel c
  //       input_tensor_values.insert(
  //           input_tensor_values.end(),
  //           (float *)channels[c].data,
  //           (float *)channels[c].data + height_resize * width_resize);
  //     }
  //
  //     // Create memory_info for Ort session
  //     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
  //     OrtMemTypeDefault); Ort::Value input_tensor =
  //     Ort::Value::CreateTensor<float>(
  //         memory_info, input_tensor_values.data(), input_tensor_size,
  //         input_shape.data(), input_shape.size());
  //
  //     Ort::AllocatorWithDefaultOptions allocator;
  //
  //     std::vector<std::string> input_names = ort_session->GetInputNames();
  //     std::vector<const char *> input_name_ptrs;
  //     for (auto &name : input_names)
  //       input_name_ptrs.push_back(name.c_str());
  //
  //     std::vector<std::string> output_names = ort_session->GetOutputNames();
  //     std::vector<const char *> output_name_ptrs;
  //     for (auto &name : output_names)
  //       output_name_ptrs.push_back(name.c_str());
  //
  //     auto output_tensors = ort_session->Run(
  //         Ort::RunOptions{nullptr},
  //         input_name_ptrs.data(),
  //         &input_tensor,
  //         1,
  //         output_name_ptrs.data(),
  //         1);
  //
  //     // Process output tensor and extract bounding boxes
  //     auto &output_tensor = output_tensors[0];
  //     auto output_shape =
  //     output_tensor.GetTensorTypeAndShapeInfo().GetShape(); float
  //     *output_data = output_tensor.GetTensorMutableData<float>();
  //
  //     auto processed_data = processDetections(rt, output_data, output_shape,
  //     static_cast<long>(height_resize), static_cast<long>(width_resize),
  //     0.7f, 0.7f);
  //
  //     auto end_time = std::chrono::high_resolution_clock::now();
  //     auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
  //     start_time); logToJSConsole(rt, std::format("Prediction completed in:
  //     {} ms.", std::to_string(duration.count())));
  //
  //     return jsi::Value(rt, std::move(processed_data));
  //   }
  //   catch (const cv::Exception &e)
  //   {
  //     throw jsi::JSError(rt, "OpenCV error: " + std::string(e.what()));
  //   }
  //   catch (const Ort::Exception &e)
  //   {
  //     throw jsi::JSError(rt, "ONNX Runtime error: " + std::string(e.what()));
  //   }
  //   catch (const std::exception &e)
  //   {
  //     throw jsi::JSError(rt, std::string("Error: ") + e.what());
  //   }
  // };
  //
  // auto jsiFunc = jsi::Function::createFromHostFunction(
  //     rt, jsi::PropNameID::forUtf8(rt, "myPlugin"), 1, myPlugin);
  //
  // rt.global().setProperty(rt, "cppPlugin", jsiFunc);
  auto myPlugin = [this](jsi::Runtime &rt, const jsi::Value &thisArgs,
                         const jsi::Value *args, size_t count) -> jsi::Value {
    try {
      auto _frame = get_frame(rt, args, count);

      [[maybe_unused]] auto width =
          _frame->get(rt, jsi::PropNameID::forUtf8(rt, "width")).asNumber();
      [[maybe_unused]] auto height =
          _frame->get(rt, jsi::PropNameID::forUtf8(rt, "height")).asNumber();
      auto pixel_format =
          _frame->get(rt, jsi::PropNameID::forUtf8(rt, "pixelFormat"))
              .asString(rt)
              .utf8(rt);
      auto get_native_buffer_fn =
          _frame->get(rt, jsi::PropNameID::forUtf8(rt, "getNativeBuffer"))
              .asObject(rt)
              .asFunction(rt);

      // Get buffer native store it as pointer in uint8
      auto buffer_obj =
          get_native_buffer_fn
              .call(rt, jsi::Object::createFromHostObject(rt, _frame))
              .asObject(rt);
      auto pointer_bigInt = buffer_obj.getProperty(rt, "pointer").asBigInt(rt);
      uintptr_t pointer = pointer_bigInt.getUint64(rt);
      AHardwareBuffer *hardware_buffer =
          reinterpret_cast<AHardwareBuffer *>(pointer);

      // Get buffer description
      AHardwareBuffer_Desc buffer_description;
      AHardwareBuffer_describe(hardware_buffer, &buffer_description);

      // Lock buffer for reading
      void *buffer;
      int result = AHardwareBuffer_lock(hardware_buffer,
                                        AHARDWAREBUFFER_USAGE_CPU_READ_MASK, -1,
                                        nullptr, &buffer);
      if (result != 0) {
        throw jsi::JSError(rt, "Failed to lock hardware buffer for reading!.");
      }

      // Pre-calculate dimensions
      constexpr int64_t width_resize = 640;
      constexpr int64_t height_resize = 640;
      constexpr size_t input_tensor_size = 1 * 3 * 640 * 640;

      cv::Mat image;
      if (buffer_description.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) {
        cv::Mat rgba(buffer_description.height, buffer_description.width,
                     CV_8UC4, buffer);

        // Direct resize from RGBA to avoid extra copy
        cv::Mat rgba_resized;
        cv::resize(rgba, rgba_resized, cv::Size(width_resize, height_resize), 0,
                   0, cv::INTER_LINEAR);

        // Convert to RGB and normalize in one step
        rgba_resized.convertTo(image, CV_32FC3, 1.0 / 255.0);
        cv::cvtColor(image, image, cv::COLOR_RGBA2RGB);
      } else if (buffer_description.format ==
                 AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420) {
        cv::Mat gray(buffer_description.height, buffer_description.width,
                     CV_8UC1);

        for (uint32_t y = 0; y < buffer_description.height; y++) {
          std::memcpy(gray.data + (y * buffer_description.width),
                      (uint8_t *)buffer + (y * buffer_description.stride),
                      buffer_description.width);
        }

        // Resize grayscale before color conversion
        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, cv::Size(width_resize, height_resize), 0,
                   0, cv::INTER_LINEAR);

        // Convert and normalize
        gray_resized.convertTo(image, CV_32FC1, 1.0 / 255.0);
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
      } else {
        cv::Mat gray(buffer_description.height, buffer_description.width,
                     CV_8UC1);
        for (uint32_t y = 0; y < buffer_description.height; y++) {
          std::memcpy(gray.data + (y * buffer_description.width),
                      (uint8_t *)buffer + (y * buffer_description.stride),
                      buffer_description.width);
        }

        // Resize grayscale before color conversion
        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, cv::Size(width_resize, height_resize), 0,
                   0, cv::INTER_LINEAR);

        // Convert and normalize
        gray_resized.convertTo(image, CV_32FC1, 1.0 / 255.0);
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
      }

      // Unlock buffer early
      AHardwareBuffer_unlock(hardware_buffer, nullptr);
      auto deleteFn =
          buffer_obj.getProperty(rt, "delete").asObject(rt).asFunction(rt);
      deleteFn.call(rt);

      // Start inference time calculation
      auto start_time = std::chrono::high_resolution_clock::now();

      // Pre-allocate input tensor with reserved capacity
      std::vector<float> input_tensor_values;
      input_tensor_values.reserve(input_tensor_size);

      // Split and insert channels directly
      cv::Mat channels[3];
      cv::split(image, channels);

      for (int c = 0; c < 3; ++c) {
        input_tensor_values.insert(input_tensor_values.end(),
                                   (float *)channels[c].datastart,
                                   (float *)channels[c].dataend);
      }

      // Create input tensor with static shape
      static const std::vector<int64_t> input_shape = {1, 3, height_resize,
                                                       width_resize};
      static auto memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_size,
          input_shape.data(), input_shape.size());

      // Cache input/output names (move to initialization if possible)
      // Cache input/output names with lazy initialization
      static std::vector<std::string> input_names_cache;
      static std::vector<const char *> input_name_ptrs;
      static std::vector<std::string> output_names_cache;
      static std::vector<const char *> output_name_ptrs;
      static bool names_initialized = false;

      if (!names_initialized) {
        Ort::AllocatorWithDefaultOptions allocator;
        input_names_cache = ort_session->GetInputNames();
        input_name_ptrs.reserve(input_names_cache.size());
        for (auto &name : input_names_cache)
          input_name_ptrs.push_back(name.c_str());

        output_names_cache = ort_session->GetOutputNames();
        output_name_ptrs.reserve(output_names_cache.size());
        for (auto &name : output_names_cache)
          output_name_ptrs.push_back(name.c_str());

        names_initialized = true;
      }

      auto output_tensors =
          ort_session->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(),
                           &input_tensor, 1, output_name_ptrs.data(), 1);

      // Process output tensor
      float *output_data = output_tensors[0].GetTensorMutableData<float>();
      auto output_shape =
          output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

      auto processed_data =
          processDetections(rt, output_data, output_shape, height_resize,
                            width_resize, 0.7f, 0.7f);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);
      logToJSConsole(rt, std::format("Prediction completed in: {} ms.",
                                     std::to_string(duration.count())));

      return jsi::Value(rt, std::move(processed_data));
    } catch (const cv::Exception &e) {
      throw jsi::JSError(rt, "OpenCV error: " + std::string(e.what()));
    } catch (const Ort::Exception &e) {
      throw jsi::JSError(rt, "ONNX Runtime error: " + std::string(e.what()));
    } catch (const std::exception &e) {
      throw jsi::JSError(rt, std::string("Error: ") + e.what());
    }
  };

  auto jsiFunc = jsi::Function::createFromHostFunction(
      rt, jsi::PropNameID::forUtf8(rt, "myPlugin"), 1, myPlugin);

  rt.global().setProperty(rt, "cppPlugin", jsiFunc);
}

void logToJSConsole(jsi::Runtime &rt, const std::string &message) {
  auto global = rt.global();

  auto console = global.getProperty(rt, "console").asObject(rt);

  auto log = console.getProperty(rt, "log").asObject(rt).asFunction(rt);

  auto jsiMessage = jsi::String::createFromUtf8(rt, "[CPP] " + message);

  log.call(rt, jsiMessage);
}

std::string matToString(const cv::Mat &mat) {
  std::ostringstream oss;
  oss << "cv::Mat("
      << "rows=" << mat.rows << ", cols=" << mat.cols
      << ", channels=" << mat.channels() << ", type=" << mat.type() << ")";
  return oss.str();
}

cv::Mat load_image(jsi::Runtime &rt, const std::string &path,
                   const std::array<int, 4> &dims) {
  std::string normalized_path = normalize_pathname(path);

  try {
    cv::Mat image = cv::imread(normalized_path, cv::IMREAD_COLOR_RGB);

    if (image.empty()) {
      logToJSConsole(rt, "Empty or failed to load image");
    } else {
      logToJSConsole(rt, "Success loading image");
    }

    cv::resize(image, image, cv::Size(dims[2], dims[3]));
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    logToJSConsole(rt, matToString(image));

    return image;
  } catch (const cv::Exception &e) {
    logToJSConsole(rt, e.what());
    return cv::Mat();
  }
}

std::string normalize_pathname(const std::string &path) // const is non-mutable
{
  std::string path_copy = path;
  path_copy.replace(0, 7, "");
  return path_copy;
}

float calculateIoU(const Detection &a, const Detection &b) {
  float x1 = std::max(a.x1(), b.x1());
  float y1 = std::max(a.y1(), b.y1());
  float x2 = std::min(a.x2(), b.x2());
  float y2 = std::min(a.y2(), b.y2());

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float areaA = a.width * a.height;
  float areaB = b.width * b.height;
  float unionArea = areaA + areaB - intersection;

  return unionArea > 0 ? intersection / unionArea : 0;
}

std::vector<Detection> performNMS(std::vector<Detection> &detections,
                                  float iouThreshold) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); i++) {
    if (suppressed[i])
      continue;

    result.push_back(detections[i]);

    for (size_t j = i + 1; j < detections.size(); j++) {
      if (suppressed[j])
        continue;

      if (detections[i].classId == detections[j].classId) {
        float iou = calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  return result;
}

std::vector<Detection> parseYOLOOutput(float *output_data,
                                       const std::vector<int64_t> &output_shape,
                                       int original_width, int original_height,
                                       float confidenceThreshold = 0.25f) {
  std::vector<Detection> detections;

  int numPredictions = output_shape[2];

  // Calculate scale factors to convert back to original image coordinates
  float scale_x = static_cast<float>(original_width) / 640.0f;
  float scale_y = static_cast<float>(original_height) / 640.0f;

  float maxConf = -999.0f;
  [[maybe_unused]] int maxIdx = -1;

  for (int i = 0; i < numPredictions; i++) {
    // Transposed format: data[feature_idx * num_predictions + prediction_idx]
    float x_center = output_data[0 * numPredictions + i];
    float y_center = output_data[1 * numPredictions + i];
    float width = output_data[2 * numPredictions + i];
    float height = output_data[3 * numPredictions + i];
    float confidence = output_data[4 * numPredictions + i];

    if (confidence > maxConf) {
      maxConf = confidence;
      maxIdx = i;
    }

    if (confidence < confidenceThreshold)
      continue;

    Detection det;
    // Scale coordinates back to original image size
    det.x = x_center * scale_x;
    det.y = y_center * scale_y;
    det.width = width * scale_x;
    det.height = height * scale_y;
    det.confidence = confidence;
    det.classId = 0;
    detections.push_back(det);
  }

  return detections;
}

jsi::Array detectionsToJSIArray(jsi::Runtime &runtime,
                                const std::vector<Detection> &detections) {
  jsi::Array result(runtime, detections.size());

  for (size_t i = 0; i < detections.size(); i++) {
    jsi::Object obj(runtime);
    obj.setProperty(runtime, "x", jsi::Value(detections[i].x));
    obj.setProperty(runtime, "y", jsi::Value(detections[i].y));
    obj.setProperty(runtime, "width", jsi::Value(detections[i].width));
    obj.setProperty(runtime, "height", jsi::Value(detections[i].height));
    obj.setProperty(runtime, "confidence",
                    jsi::Value(detections[i].confidence));
    obj.setProperty(runtime, "classId", jsi::Value(detections[i].classId));

    result.setValueAtIndex(runtime, i, std::move(obj));
  }

  return result;
}

jsi::Array processDetections(jsi::Runtime &runtime, float *output_data,
                             const std::vector<int64_t> &output_shape,
                             int original_width, int original_height,
                             float confidenceThreshold = 0.25f,
                             float iouThreshold = 0.45f) {
  auto detections = parseYOLOOutput(output_data, output_shape, original_width,
                                    original_height, confidenceThreshold);
  auto filteredDetections = performNMS(detections, iouThreshold);
  return detectionsToJSIArray(runtime, filteredDetections);
}

std::shared_ptr<vision::FrameHostObject>
get_frame(jsi::Runtime &rt, const jsi::Value *args, size_t count) {
  if (count == 0 || !args[0].isObject()) {
    throw jsi::JSError(rt, "Invalid arguments");
  }

  auto obj = args[0].asObject(rt);
  if (!obj.isHostObject(rt)) {
    throw jsi::JSError(rt, "Not a HostObject");
  }

  auto host_object = obj.getHostObject(rt);
  auto _frame = std::static_pointer_cast<vision::FrameHostObject>(host_object);
  if (!_frame) {
    throw jsi::JSError(rt, "Object is not a FrameHostObject");
  }

  return _frame;
}

} // namespace facebook::react
