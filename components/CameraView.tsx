import { StyleSheet, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useSkiaFrameProcessor,
} from 'react-native-vision-camera';
import CustomButton from './Button';
import { Dispatch, SetStateAction } from 'react';
import { useSharedValue } from 'react-native-reanimated';
import { PaintStyle, Skia } from '@shopify/react-native-skia';

type CameraViewProps = {
  isActive: boolean;
  onPress: Dispatch<SetStateAction<boolean>>;
};

const CameraView = ({ isActive, onPress }: Readonly<CameraViewProps>) => {
  const device = useCameraDevice('back');

  const lastProcessed = useSharedValue(0);
  const detections = useSharedValue([]);

  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  paint.setStyle(PaintStyle.Stroke);
  paint.setStrokeWidth(2);

  const frameProcessor = useSkiaFrameProcessor(frame => {
    'worklet';

    if (frame.pixelFormat !== 'rgb') return;
    frame.render();

    // Throttle detection
    const currentTime = Date.now();
    const interval = 300;

    if (currentTime - lastProcessed.value >= interval) {
      lastProcessed.value = currentTime;
      const result = cppPlugin(frame) as any;
      detections.value = result;
    }

    const dets = detections.value;
    if (dets && dets.length > 0) {
      const frameWidth = frame.width;
      const frameHeight = frame.height;

      const modelWidth = 640;
      const modelHeight = 640;

      const scaleX = frameWidth / modelWidth;
      const scaleY = frameHeight / modelHeight;

      const len = dets.length;
      for (let i = 0; i < len; i++) {
        const d = dets[i] as any;

        const x = d.x * scaleX;
        const y = d.y * scaleY;
        const width = d.width * scaleX;
        const height = d.height * scaleY;

        frame.drawRect(
          Skia.XYWHRect(x - width / 2, y - height / 2, width, height),
          paint,
        );
      }
    }
  }, []);

  if (!device) return null;

  const onPressCloseCamera = () => {
    onPress(false);
  };

  return (
    <View style={StyleSheet.absoluteFillObject}>
      <Camera
        style={StyleSheet.absoluteFillObject}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        pixelFormat={'rgb'}
        enableFpsGraph={true}
        enableZoomGesture={true}
      />
      <CustomButton
        style={[StyleSheet.absoluteFillObject, styles.buttonContainer]}
        label={'back'}
        onPress={onPressCloseCamera}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  buttonContainer: {
    padding: 36,
  },
});

export default CameraView;
