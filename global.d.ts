import { Frame } from "react-native-vision-camera";

declare global {
  // Use 'var' to declare a property on the global scope.
  const cppPlugin: (frame: Frame) => number;
}

declare module '@shopify/react-native-skia';

// Adding this empty export statement turns the file into a module.
export {};