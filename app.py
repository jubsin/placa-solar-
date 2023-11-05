from flask import Flask, request, render_template, jsonify
from keras.optimizers import Adamax
import paho.mqtt.publish as publish
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado de IA
modelo = tf.keras.models.load_model(r"C:\Users\julia\Downloads\Solar Panel dust (0.90).h5")
modelo.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

@app.route('/')
def index():
    return render_template('index_2.html')

# Configurações do MQTT Broker
mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883
mqtt_topic = "esp32/topic"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        #imagem = request.files['imagem']
        arquivo = request.files['imagem']

        # Processar a imagem usando o modelo de IA
        img = Image.open(arquivo).resize((224, 224))  # Ajuste o tamanho conforme necessário para o seu modelo
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        previsao = modelo.predict(img_array)
        classe_predita = tf.argmax(previsao[0])
        # Mapear o índice da classe para a saída desejada
        if classe_predita == 0:
            resultado = "Clean"
        else:
            resultado = "Dusty"

        # Enviar a resposta para o ESP32 via MQTT
        publish.single(mqtt_topic, resultado, hostname=mqtt_broker, port=mqtt_port)
        print(resultado)
        return jsonify({"resultado": resultado})
    except Exception as e:
        return f'Erro: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
