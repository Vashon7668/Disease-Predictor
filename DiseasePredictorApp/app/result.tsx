import React from "react";
import { View, Text, StyleSheet, Button } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

export default function ResultScreen() {
  const router = useRouter();
  const { result, token, wait_time } = useLocalSearchParams();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Prediction Result</Text>
      <Text style={styles.label}>Result:</Text>
      <Text style={styles.value}>{result}</Text>

      <Text style={styles.label}>Token:</Text>
      <Text style={styles.value}>{token}</Text>

      <Text style={styles.label}>Estimated Wait Time:</Text>
      <Text style={styles.value}>{wait_time} mins</Text>

      <View style={styles.buttonContainer}>
        <Button title="Back to Home" onPress={() => router.replace("/")} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    justifyContent: "center",
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
  },
  label: {
    fontSize: 18,
    marginTop: 10,
    fontWeight: "600",
  },
  value: {
    fontSize: 16,
    color: "#333",
    marginBottom: 10,
  },
  buttonContainer: {
    marginTop: 30,
  },
});
