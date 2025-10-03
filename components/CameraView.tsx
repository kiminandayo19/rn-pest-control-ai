import { StyleSheet, View } from 'react-native';
import {
  Camera,
  Frame,
  useCameraDevice,
  useSkiaFrameProcessor,
} from 'react-native-vision-camera';
import CustomButton from './Button';
import { Dispatch, SetStateAction } from 'react';
import { useSharedValue } from 'react-native-reanimated';
import { PaintStyle, Skia } from '@shopify/react-native-skia';
import { runOnJS } from 'react-native-worklets';

type CameraViewProps = {
  isActive: boolean;
  onPress: Dispatch<SetStateAction<boolean>>;
};

const CameraView = ({ isActive, onPress }: Readonly<CameraViewProps>) => {
  const device = useCameraDevice('back');

  const lastProcessed = useSharedValue(0);
  const detections = useSharedValue([]);
  const isProcessing = useSharedValue(false);

  const paint = Skia.Paint();
  paint.setColor(Skia.Color('red'));
  paint.setStyle(PaintStyle.Stroke);
  paint.setStrokeWidth(2);

  const frameProcessor = useSkiaFrameProcessor(frame => {
    'worklet';

    if (frame.pixelFormat !== 'rgb') return;

    // Render frame first
    frame.render();

    const dets = detections.value;
    if (dets && dets.length > 0) {
      const scaleX = frame.width / 640;
      const scaleY = frame.height / 640;

      for (let i = 0; i < dets.length; i++) {
        const d = dets[i] as any;

        const scaledX = d.x * scaleX;
        const scaledY = d.y * scaleY;
        const scaledWidth = d.width * scaleX;
        const scaledHeight = d.height * scaleY;

        frame.drawRect(
          Skia.XYWHRect(
            scaledX - scaledWidth * 0.5,
            scaledY - scaledHeight * 0.5,
            scaledWidth,
            scaledHeight,
          ),
          paint,
        );
      }
    }

    // Only process if not already processing
    const currentTime = Date.now();
    if (currentTime - lastProcessed.value >= 500 && !isProcessing.value) {
      lastProcessed.value = currentTime;
      isProcessing.value = true;

      const result = cppPlugin(frame);
      detections.value = result;
      isProcessing.value = false;
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
