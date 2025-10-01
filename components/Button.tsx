import {
  TouchableOpacity,
  Text,
  TouchableOpacityProps,
  GestureResponderEvent,
  StyleSheet,
} from 'react-native';

interface IButtonProps<A, R>
  extends Omit<TouchableOpacityProps, 'onPress' | 'title'> {
  label: string;
  onPress: (arg0?: A | GestureResponderEvent) => R | void;
}

const CustomButton = <A, R>({
  label,
  onPress,
  ...buttonProps
}: Readonly<IButtonProps<A, R>>) => {
  return (
    <TouchableOpacity
      style={styles.buttonContainerStyle}
      onPress={onPress}
      {...buttonProps}
    >
      <Text style={styles.buttonTextStyle}>{label}</Text>
    </TouchableOpacity>
  );
};

export default CustomButton;

const styles = StyleSheet.create({
  buttonContainerStyle: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#31326F',
    borderRadius: 16,
  },
  buttonTextStyle: {
    color: '#A8FBD3',
  },
});
