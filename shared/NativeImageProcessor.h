#pragma once

#include <ImageSpecsJSI.h> // Assumes 'ImageSpecs' is your codegen name

#include <array> // Include for std::array
#include <memory>
#include <format>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <jni.h>
#include <jsi/jsi.h>
#include <chrono>
#include <android/log.h>
#include <fbjni/fbjni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/hardware_buffer.h>
#include "frameprocessors/FrameHostObject.h"

struct Detection
{
  float x, y, width, height;
  float confidence;
  int classId;

  float x1() const { return x - width / 2; }
  float y1() const { return y - height / 2; }
  float x2() const { return x + width / 2; }
  float y2() const { return y + height / 2; }
};

namespace facebook::react
{
  extern AAssetManager *g_assetManager;

  class NativeImageProcessor
      : public NativeImageProcessorCxxSpec<NativeImageProcessor>
  {
  private:
    std::unique_ptr<Ort::Env> ort_env;
    std::unique_ptr<Ort::Session> ort_session;

  public:
    explicit NativeImageProcessor(std::shared_ptr<CallInvoker> jsInvoker);

    void loadModel(jsi::Runtime &rt);

    jsi::Array predict(jsi::Runtime &rt, const std::string &path, const std::array<int, 4> &dims);
    void registerVisionCameraExtension(jsi::Runtime &rt);
  };

  void logToJSConsole(jsi::Runtime &rt, const std::string &message);

  std::string matToString(const cv::Mat &mat);

  cv::Mat load_image(jsi::Runtime &rt, const std::string &path, const std::array<int, 4> &dims);

  std::string normalize_pathname(const std::string &path);
  float calculateIoU(const Detection &a, const Detection &b);
  std::vector<Detection> performNMS(std::vector<Detection> &detections,
                                    float iouThreshold = 0.7f);
  std::vector<Detection> parseYOLOOutput(float *output_data,
                                         const std::vector<int64_t> &output_shape,
                                         float confidenceThreshold = 0.7f);
  jsi::Array detectionsToJSIArray(jsi::Runtime &runtime,
                                  const std::vector<Detection> &detections);
  jsi::Array processDetections(jsi::Runtime &runtime,
                               float *output_data,
                               const std::vector<int64_t> &output_shape,
                               int original_width,
                               int original_height,
                               float confidenceThreshold,
                               float iouThreshold);

  std::shared_ptr<vision::FrameHostObject> get_frame(jsi::Runtime &rt, const jsi::Value *args, size_t count);

} // namespace facebook::react
