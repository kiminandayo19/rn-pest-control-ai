import { Frame } from "react-native-vision-camera";
import NativeImageProcessor from "./specs/NativeImageProcessor";

let pestControlVisionCameraPlugin: (frame: Frame) => void;

if (!(globalThis as any)?.cppPlugin) {
  NativeImageProcessor.registerVisionCameraExtension();
  pestControlVisionCameraPlugin = (globalThis as any)?.cppPlugin;
  NativeImageProcessor.loadModel();
} else {
  pestControlVisionCameraPlugin = () => {};
}

export const cppPlugin = pestControlVisionCameraPlugin