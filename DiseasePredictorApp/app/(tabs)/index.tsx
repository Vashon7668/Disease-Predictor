import React, { useState } from "react";
import { View, Text, TextInput, Button, StyleSheet, ScrollView } from "react-native";
import { useRouter } from "expo-router";

export default function DiseasePredictor() {
  const router = useRouter();

  const [fever, setFever] = useState("1");
  const [cough, setCough] = useState("1");
  const [fatigue, setFatigue] = useState("1");
  const [breathing, setBreathing] = useState("1");
  const [age, setAge] = useState("18");
  const [gender, setGender] = useState("1");
  const [bloodPressure, setBloodPressure] = useState("1");
  const [cholesterol, setCholesterol] = useState("1");

  const handlePredict = async () => {
    const payload = {
      fever: parseInt(fever),
      cough: parseInt(cough),
      fatigue: parseInt(fatigue),
      breathing: parseInt(breathing),
      age: parseInt(age),
      gender: parseInt(gender),
      blood_pressure: parseInt(bloodPressure),
      cholesterol: parseInt(cholesterol),
    };

    try {
      const response = await fetch("https://bd9d-210-18-173-163.ngrok-free.app/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        router.push({
          pathname: "/result",
          params: {
            result: data.result,
            token: data.token,
            wait_time: data.wait_time.toString(),
          },
        });
      } else {
        router.push({
          pathname: "/result",
          params: {
            result: data.error || "Prediction failed.",
            token: "-",
            wait_time: "-",
          },
        });
      }
    } catch (error) {
      console.error("Error:", error);
      router.push({
        pathname: "/result",
        params: {
          result: "Could not connect to the server.",
          token: "-",
          wait_time: "-",
        },
      });
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Disease Predictor</Text>
      <TextInput style={styles.input} value={fever} onChangeText={setFever} placeholder="Fever" keyboardType="numeric" />
      <TextInput style={styles.input} value={cough} onChangeText={setCough} placeholder="Cough" keyboardType="numeric" />
      <TextInput style={styles.input} value={fatigue} onChangeText={setFatigue} placeholder="Fatigue" keyboardType="numeric" />
      <TextInput style={styles.input} value={breathing} onChangeText={setBreathing} placeholder="Breathing" keyboardType="numeric" />
      <TextInput style={styles.input} value={age} onChangeText={setAge} placeholder="Age" keyboardType="numeric" />
      <TextInput style={styles.input} value={gender} onChangeText={setGender} placeholder="Gender" keyboardType="numeric" />
      <TextInput style={styles.input} value={bloodPressure} onChangeText={setBloodPressure} placeholder="Blood Pressure" keyboardType="numeric" />
      <TextInput style={styles.input} value={cholesterol} onChangeText={setCholesterol} placeholder="Cholesterol" keyboardType="numeric" />
      <Button title="PREDICT" onPress={handlePredict} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    justifyContent: "center",
    backgroundColor: "#fff", // dark background
    flexGrow: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
    color: "#000", // white text for dark mode
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    backgroundColor: "#fff", // white input box
    color: "#000", // black text
    padding: 10,
    marginBottom: 10,
    borderRadius: 5,
  },
});

