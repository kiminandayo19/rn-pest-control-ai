import { StyleSheet, View } from 'react-native';

interface IGapProps {
  size: number;
}

const Gap = ({ size }: Readonly<IGapProps>) => {
  const style = styles(size);
  return <View style={style.gapContainer} />;
};

export default Gap;

const styles = (size: number) =>
  StyleSheet.create({
    gapContainer: {
      width: size,
      height: size,
    },
  });
