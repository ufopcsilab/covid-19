/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React from 'react';
import {SafeAreaView, Button, Text} from 'react-native';

const App: () => React$Node = () => {
  return (
    <>
      <StatusBar barStyle="dark-content" />
      <SafeAreaView>
        <Text>Insira uma imagem:</Text>

        <Button> Buscar imagem </Button>
      </SafeAreaView>
    </>
  );
};

export default App;
