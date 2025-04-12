import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, ScrollView } from 'react-native';
import axios from 'axios';

const App = () => {
  const [result, setResult] = useState(null);
  const [token, setToken] = useState(null);
  const [waitTime, setWaitTime] = useState(null);
  const [error, setError] = useState(null);

  const sendData = async () => {
    try {
      const response = await axios.post('https://bd9d-210-18-173-163.ngrok-free.app', {
        fever: 1,
        cough: 1,
        fatigue: 0,
        breathing: 1,
        age: 28,
        gender: 1,
        blood_pressure: 120,
        cholesterol: 200,
      }, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      const data = response.data;
      setResult(data.result);
      setToken(data.token);
      setWaitTime(data.wait_time);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to backend.");
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Disease Prediction</Text>
      <Button title="Send Data to Model" onPress={sendData} />

      {result && (
        <>
          <Text style={styles.text}>🧠 Result: {result}</Text>
          <Text style={styles.text}>🎟️ Token: {token}</Text>
          <Text style={styles.text}>⏱️ Wait Time: {waitTime}</Text>
        </>
      )}

      {error && <Text style={styles.error}>❌ {error}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
  padding: 20,
  paddingTop: 60,
  alignItems: 'center',
  backgroundColor: '#fff',        // <-- Add this
  minHeight: '100%',              // <-- Ensure full screen
  },
  title: {
    fontSize: 26,
    marginBottom: 30,
    fontWeight: 'bold',
  },
  input: {
    width: '100%',
    marginVertical: 10,
    padding: 10,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 8,
  },
  input: {
  width: '100%',
  marginVertical: 10,
  padding: 10,
  borderColor: '#ccc',
  borderWidth: 1,
  borderRadius: 8,
  backgroundColor: '#fff',         // <-- Add this line
  color: '#000',                   // <-- Add this line
  },
  resultContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  result: {
    fontSize: 18,
    fontWeight: '600',
    color: 'green',
  },
});


export default App;
