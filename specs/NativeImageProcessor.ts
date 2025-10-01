import { TurboModule, TurboModuleRegistry } from 'react-native';

export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  classId: number;
  score: number;
};

export interface Spec extends TurboModule {
  readonly loadModel: () => void;
  readonly predict: (path: string, dims: number[]) => Array<BoundingBox>;
  readonly registerVisionCameraExtension: () => void;
}

export default TurboModuleRegistry.getEnforcing<Spec>('NativeImageProcessor');
