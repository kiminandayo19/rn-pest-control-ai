import { useState } from 'react';
import { Alert, Dimensions, Image, StyleSheet, View } from 'react-native';
import {
  Asset,
  ImageLibraryOptions,
  launchImageLibrary,
} from 'react-native-image-picker';
import CustomButton from './Button';
import Gap from './Gap';
import NativeImageProcessor from '../specs/NativeImageProcessor';
import { Camera } from 'react-native-vision-camera';
import CameraView from './CameraView';

const ImagePickerComponent = () => {
  const [image, setImage] = useState<Asset | null>(null);
  const [isGranted, setIsGranted] = useState<boolean>(false);

  const onPressUpload = async () => {
    const imageLibraryOpts: ImageLibraryOptions = {
      mediaType: 'photo',
      quality: 0.8,
    };
    const imageResult = await launchImageLibrary(imageLibraryOpts);
    if (!imageResult?.assets?.[0]?.uri) return;
    setImage(imageResult?.assets?.[0]);
    NativeImageProcessor.predict(imageResult?.assets?.[0]?.uri, [1, 3, 640, 640]);
    
  };

  const onPressDelete = () => setImage(null);

  const onPressOpenCamera = async () => {
    const granted = await Camera.requestCameraPermission();

    if (granted === 'granted') {
      setIsGranted(true);
    } else {
      setIsGranted(false);
      Alert.alert('Apps need to access camera. Please allow');
    }
  };

  const renderImage = () => {
    if (!image?.uri) return null;

    return (
      <Image
        alt={image?.id}
        source={{ uri: image?.uri }}
        style={{
          width: image?.width,
          height: image?.height,
        }}
      />
    );
  };

  const renderCamera = () => {
    if (isGranted) {
      return <CameraView isActive={isGranted} onPress={setIsGranted} />;
    }
    return null;
  };

  return (
    <View style={styles.container}>
      <View style={styles.buttonContainer}>
        <CustomButton label={'Upload'} onPress={onPressUpload} />
        <CustomButton label={'Delete'} onPress={onPressDelete} />
        <CustomButton label={'Open Camera'} onPress={onPressOpenCamera} />
      </View>
      <Gap size={24} />
      {renderImage()}
      {renderCamera()}
    </View>
  );
};

export default ImagePickerComponent;

const styles = StyleSheet.create({
  container: {
    width: Dimensions.get('screen').width,
    height: Dimensions.get('screen').height,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 8,
  },
});
