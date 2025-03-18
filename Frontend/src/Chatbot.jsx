import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import classes from './classes.json';
import intents from './intents.json';
import vocab from './vocab.json';

const Chatbot = () => {
    const [model, setModel] = useState(null);
    const [inputText, setInputText] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const chatContainerRef = useRef(null);
    const endOfChatRef = useRef(null);

    // Cargar el modelo desde la carpeta public/model
    useEffect(() => {
        const loadModel = async () => {
            try {
                const loadedModel = await tf.loadLayersModel(`${import.meta.env.BASE_URL}model/model.json`);
                setModel(loadedModel);
                console.log('Modelo cargado');
            } catch (error) {
                console.error('Error cargando el modelo:', error);
            }
        };
        loadModel();
    }, []);

    // Auto-scroll: desplazar al final cuando chatHistory cambie
    useEffect(() => {
        setTimeout(() => {
            if (endOfChatRef.current) {
                endOfChatRef.current.scrollIntoView({ behavior: 'smooth' });
            }
        }, 100);
    }, [chatHistory]);


    // Función para preprocesar el input (bag of words)
    const preprocessInput = (text) => {
        const tokens = text.toLowerCase().split(' ');
        const vector = new Array(vocab.length).fill(0);
        tokens.forEach(token => {
            const index = vocab.indexOf(token);
            if (index !== -1) {
                vector[index] = 1;
            }
        });
        return tf.tensor2d([vector], [1, vocab.length]);
    };

    // Mapear la predicción al tag y obtener respuesta
    const mapPredictionToTag = (predictedIndex) => classes[predictedIndex];
    const getResponseForTag = (tag) => {
        const intent = intents.intents.find(intent => intent.tag === tag);
        if (intent) {
            const randomIndex = Math.floor(Math.random() * intent.responses.length);
            if (intent.image_url) {
                return {
                    tipo: "imagen",
                    texto: intent.responses[randomIndex],
                    image_url: intent.image_url
                };
            } else {
                return {
                    tipo: "texto",
                    texto: intent.responses[randomIndex]
                };
            }
        }
        return { tipo: "texto", texto: "No entiendo." };
    };

    // Manejar el envío del mensaje
    const handleSend = async () => {
        if (!model) {
            console.log('El modelo aún no está cargado.');
            return;
        }
        const inputTensor = preprocessInput(inputText);
        const predictionTensor = model.predict(inputTensor);
        const predictionArray = predictionTensor.dataSync();
        const predictedIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const tag = mapPredictionToTag(predictedIndex);
        const botResponse = getResponseForTag(tag);

        setChatHistory([...chatHistory, { user: inputText, bot: botResponse }]);
        setInputText('');
    };

    const userBubbleStyle = {
        backgroundColor: '#0084FF',
        color: '#fff',
        padding: '10px 15px',
        borderRadius: '20px',
        maxWidth: '70%',
        alignSelf: 'flex-end',
        marginBottom: '10px'
    };

    const botBubbleStyle = {
        backgroundColor: '#CCE1FF',
        color: '#000',
        padding: '10px 15px',
        borderRadius: '20px',
        maxWidth: '70%',
        alignSelf: 'flex-start',
        marginBottom: '10px'
    };

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: '20px',
                fontFamily: 'Arial, sans-serif',
                backgroundColor: '#202020'
            }}
        >
            <h2 style={{ textAlign: 'center', color: '#fff' }}>Chatbot</h2>
            <div
                ref={chatContainerRef}
                style={{
                    border: '1px solid #ccc',
                    borderRadius: '10px',
                    padding: '10px',
                    marginBottom: '20px',
                    height: '600px',
                    width: '500px',
                    overflowY: 'auto',
                    backgroundColor: '#f9f9f9'
                }}
            >
                {chatHistory.map((entry, index) => (
                    <div key={index} style={{ display: 'flex', flexDirection: 'column' }}>
                        <div style={userBubbleStyle}>
                            <strong>Tú:</strong> {entry.user}
                        </div>
                        {entry.bot.tipo === 'imagen' ? (
                            <div style={botBubbleStyle}>
                                <div>
                                    <strong>Bot:</strong> {entry.bot.texto}
                                </div>
                                <img
                                    src={entry.bot.image_url}
                                    alt="Imagen del bot"
                                    style={{
                                        marginTop: '5px',
                                        maxWidth: '100%',
                                        maxHeight: '300px',
                                        borderRadius: '10px',
                                        objectFit: 'contain'
                                    }}
                                />
                            </div>
                        ) : (
                            <div style={botBubbleStyle}>
                                <strong>Bot:</strong> {entry.bot.texto}
                            </div>
                        )}
                    </div>
                ))}
                <div ref={endOfChatRef} />
            </div>
            <div
                style={{
                    display: 'flex',
                    width: '100%',
                    maxWidth: '800px',
                    margin: '20px auto 0 auto'
                }}
            >
                <input
                    type="text"
                    placeholder="Escribe tu mensaje..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
                    style={{
                        flex: 1,
                        padding: '8px',
                        borderRadius: '5px',
                        border: '1px solid #ccc',
                        color: '#fff'
                    }}
                />
                <button
                    onClick={handleSend}
                    style={{
                        padding: '8px 16px',
                        marginLeft: '10px',
                        borderRadius: '5px',
                        border: 'none',
                        backgroundColor: '#007bff',
                        color: 'white',
                        cursor: 'pointer'
                    }}
                >
                    Enviar
                </button>
            </div>
        </div>
    );
};

export default Chatbot;
