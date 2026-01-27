import { useState, useEffect, useRef } from 'react';
import './SymptomChecker.css';

const SymptomChecker = () => {
  const [messages, setMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [userInput, setUserInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatStarted, setChatStarted] = useState(false);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const messagesEndRef = useRef(null);

  const API_URL = 'http://localhost:5000';

  const questions = [
    {
      id: 'problem',
      text: 'What symptoms are you experiencing? (Select all that apply)',
      type: 'multiple',
      options: [], // Will be populated with symptoms from API
    },
    {
      id: 'duration',
      text: 'Since when are you having these symptoms?',
      type: 'buttons',
      options: ['Today', '2-3 days', '1 week', '2 weeks', 'More than a month'],
    },
    {
      id: 'onset',
      text: 'Did it start suddenly or gradually?',
      type: 'buttons',
      options: ['Suddenly', 'Gradually'],
    },
    {
      id: 'frequency',
      text: 'Is it constant or does it come and go?',
      type: 'buttons',
      options: ['Constant', 'Comes and goes'],
    },
    {
      id: 'recurrence',
      text: 'Has it happened before?',
      type: 'buttons',
      options: ['Yes', 'No'],
    },
    {
      id: 'severity',
      text: 'How severe is the pain?',
      type: 'buttons',
      options: ['Mild', 'Moderate', 'Severe', 'No Pain'],
    },
    {
      id: 'timing',
      text: 'Is the pain more at a particular time?',
      type: 'buttons',
      options: ['Morning', 'Night', 'No specific time'],
    },
    {
      id: 'fever',
      text: 'Do you have fever?',
      type: 'buttons',
      options: ['Yes', 'No'],
    },
    {
      id: 'nausea',
      text: 'Any nausea or vomiting?',
      type: 'buttons',
      options: ['Yes', 'No'],
    },
    {
      id: 'medical_conditions',
      text: 'Do you have any existing medical conditions?',
      type: 'buttons',
      options: ['Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'None'],
    },
    {
      id: 'lifestyle',
      text: 'Do you smoke or drink alcohol?',
      type: 'buttons',
      options: ['Smoke', 'Drink', 'Both', 'None'],
    },
  ];

  useEffect(() => {
    fetchSymptoms();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSymptoms = async () => {
    try {
      const response = await fetch(`${API_URL}/api/symptoms`);
      const data = await response.json();
      setAvailableSymptoms(data.symptoms);
    } catch (err) {
      console.error('Error fetching symptoms:', err);
    }
  };

  const startChat = () => {
    setChatStarted(true);
    const welcomeMessage = {
      sender: 'ai',
      text: '👋 Hello! I\'m your AI Health Assistant. I\'ll ask you a few questions to understand your condition better.',
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
    
    setTimeout(() => {
      askQuestion(0);
    }, 1000);
  };

  const askQuestion = (questionIndex) => {
    if (questionIndex < questions.length) {
      const question = questions[questionIndex];
      
      // For the first question, populate with symptoms from API
      if (question.id === 'problem') {
        question.options = availableSymptoms;
      }
      
      const newMessage = {
        sender: 'ai',
        text: question.text,
        type: question.type,
        options: question.options,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, newMessage]);
    } else {
      // All questions answered, analyze and predict
      analyzeSymptomsAndPredict();
    }
  };

  const handleUserResponse = (answer) => {
    const question = questions[currentQuestion];
    
    // Add user message
    const userMessage = {
      sender: 'user',
      text: answer,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Store answer
    setUserAnswers(prev => ({
      ...prev,
      [question.id]: answer,
    }));

    setUserInput('');
    setSelectedSymptoms([]);
    setCurrentQuestion(prev => prev + 1);

    // Ask next question after a delay
    setTimeout(() => {
      askQuestion(currentQuestion + 1);
    }, 800);
  };

  const handleSymptomToggle = (symptom) => {
    setSelectedSymptoms(prev => {
      if (prev.includes(symptom)) {
        return prev.filter(s => s !== symptom);
      } else {
        return [...prev, symptom];
      }
    });
  };

  const handleSymptomConfirm = () => {
    if (selectedSymptoms.length > 0) {
      const formattedAnswer = selectedSymptoms.join(', ');
      handleUserResponse(formattedAnswer);
      setSearchTerm(''); // Reset search after confirming
    }
  };

  const formatSymptomName = (symptom) => {
    return symptom.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const filterSymptoms = (symptoms, search) => {
    if (!symptoms || !Array.isArray(symptoms)) return [];
    if (!search.trim()) return symptoms;
    
    const searchLower = search.toLowerCase();
    return symptoms.filter(symptom => {
      const formattedName = formatSymptomName(symptom).toLowerCase();
      const originalName = symptom.toLowerCase();
      return formattedName.includes(searchLower) || originalName.includes(searchLower);
    });
  };

  const mapAnswersToSymptoms = (answers) => {
    const symptoms = {};
    const influencingFactors = []; // Track what influenced the prediction
    
    // Initialize all symptoms as false
    availableSymptoms.forEach(symptom => {
      symptoms[symptom] = false;
    });

    // 1. Use selected symptoms from the first question (PRIMARY)
    if (answers.problem) {
      const selectedSymptomList = answers.problem.split(', ');
      selectedSymptomList.forEach(symptom => {
        if (availableSymptoms.includes(symptom)) {
          symptoms[symptom] = true;
        }
      });
      influencingFactors.push(`Selected symptoms: ${selectedSymptomList.length} symptoms`);
    }

    // 2. Fever question mapping (enhanced)
    if (answers.fever === 'Yes') {
      const feverSymptoms = availableSymptoms.filter(s => 
        s.toLowerCase().includes('fever') || s.toLowerCase().includes('chills') || 
        s.toLowerCase().includes('hot flashes') || s.toLowerCase().includes('sweating') ||
        s.toLowerCase().includes('temperature')
      );
      feverSymptoms.forEach(s => symptoms[s] = true);
      if (feverSymptoms.length > 0) {
        influencingFactors.push(`Added ${feverSymptoms.length} fever-related symptoms`);
      }
    }

    // 3. Nausea/Vomiting question mapping (enhanced)
    if (answers.nausea === 'Yes') {
      const nauseaSymptoms = availableSymptoms.filter(s => 
        s.toLowerCase().includes('nausea') || s.toLowerCase().includes('vomiting') ||
        s.toLowerCase().includes('stomach') || s.toLowerCase().includes('abdominal') ||
        s.toLowerCase().includes('gastrointestinal')
      );
      nauseaSymptoms.forEach(s => symptoms[s] = true);
      if (nauseaSymptoms.length > 0) {
        influencingFactors.push(`Added ${nauseaSymptoms.length} nausea/digestive symptoms`);
      }
    }

    // 4. Severity mapping - affects pain-related symptoms
    if (answers.severity) {
      const painSymptoms = availableSymptoms.filter(s => 
        s.toLowerCase().includes('pain') || s.toLowerCase().includes('ache') ||
        s.toLowerCase().includes('sore') || s.toLowerCase().includes('tender')
      );
      
      if (answers.severity === 'Severe') {
        painSymptoms.forEach(s => symptoms[s] = true);
        influencingFactors.push(`Severe pain - enhanced pain symptoms (${painSymptoms.length} added)`);
      } else if (answers.severity === 'Moderate' && painSymptoms.length > 0) {
        // Select more pain symptoms for moderate
        painSymptoms.slice(0, Math.ceil(painSymptoms.length * 0.7)).forEach(s => symptoms[s] = true);
        influencingFactors.push(`Moderate pain - added ${Math.ceil(painSymptoms.length * 0.7)} pain symptoms`);
      } else if (answers.severity === 'Mild' && painSymptoms.length > 0) {
        painSymptoms.slice(0, Math.ceil(painSymptoms.length * 0.3)).forEach(s => symptoms[s] = true);
        influencingFactors.push(`Mild pain - added ${Math.ceil(painSymptoms.length * 0.3)} pain symptoms`);
      }
    }

    // 5. Duration mapping - chronic vs acute (more aggressive)
    if (answers.duration) {
      if (answers.duration === 'More than a month' || answers.duration === '2 weeks') {
        // Chronic symptoms - add more fatigue and weakness symptoms
        const chronicSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('chronic') || s.toLowerCase().includes('fatigue') ||
          s.toLowerCase().includes('weakness') || s.toLowerCase().includes('tired') ||
          s.toLowerCase().includes('malaise') || s.toLowerCase().includes('low energy')
        );
        chronicSymptoms.forEach(s => symptoms[s] = true);
        if (chronicSymptoms.length > 0) {
          influencingFactors.push(`Long duration (${answers.duration}) - added ${chronicSymptoms.length} chronic symptoms`);
        }
      } else if (answers.duration === 'Today' || answers.duration === '2-3 days') {
        // Acute symptoms - be more inclusive
        const acuteSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('acute') || s.toLowerCase().includes('sudden') ||
          s.toLowerCase().includes('sharp')
        );
        acuteSymptoms.forEach(s => symptoms[s] = true);
        if (acuteSymptoms.length > 0) {
          influencingFactors.push(`Short duration (${answers.duration}) - acute presentation (${acuteSymptoms.length} added)`);
        }
      }
    }

    // 6. Onset mapping
    if (answers.onset === 'Suddenly') {
      const suddenSymptoms = availableSymptoms.filter(s =>
        s.toLowerCase().includes('sharp') || s.toLowerCase().includes('acute')
      );
      suddenSymptoms.forEach(s => symptoms[s] = true);
      if (suddenSymptoms.length > 0) {
        influencingFactors.push('Sudden onset - considered acute conditions');
      }
    }

    // 7. Timing mapping
    if (answers.timing === 'Morning') {
      const morningSymptoms = availableSymptoms.filter(s =>
        s.toLowerCase().includes('stiffness') || s.toLowerCase().includes('joint')
      );
      morningSymptoms.forEach(s => symptoms[s] = true);
      if (morningSymptoms.length > 0) {
        influencingFactors.push('Morning symptoms - added stiffness/joint issues');
      }
    } else if (answers.timing === 'Night') {
      const nightSymptoms = availableSymptoms.filter(s =>
        s.toLowerCase().includes('night') || s.toLowerCase().includes('restless') ||
        s.toLowerCase().includes('insomnia')
      );
      nightSymptoms.forEach(s => symptoms[s] = true);
      if (nightSymptoms.length > 0) {
        influencingFactors.push('Night symptoms - considered sleep-related issues');
      }
    }

    // 8. Medical conditions mapping
    if (answers.medical_conditions) {
      if (answers.medical_conditions === 'Diabetes') {
        const diabetesSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('urin') || s.toLowerCase().includes('thirst') ||
          s.toLowerCase().includes('weight')
        );
        diabetesSymptoms.forEach(s => symptoms[s] = true);
        influencingFactors.push('Diabetes history - considered related complications');
      } else if (answers.medical_conditions === 'Hypertension') {
        const hyperSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('headache') || s.toLowerCase().includes('dizz')
        );
        hyperSymptoms.forEach(s => symptoms[s] = true);
        influencingFactors.push('Hypertension history - considered related symptoms');
      } else if (answers.medical_conditions === 'Heart Disease') {
        const heartSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('chest') || s.toLowerCase().includes('breath') ||
          s.toLowerCase().includes('palpitation')
        );
        heartSymptoms.forEach(s => symptoms[s] = true);
        influencingFactors.push('Heart disease history - considered cardiac symptoms');
      } else if (answers.medical_conditions === 'Asthma') {
        const asthmaSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('breath') || s.toLowerCase().includes('wheez') ||
          s.toLowerCase().includes('cough')
        );
        asthmaSymptoms.forEach(s => symptoms[s] = true);
        influencingFactors.push('Asthma history - considered respiratory symptoms');
      }
    }

    // 9. Lifestyle factors mapping
    if (answers.lifestyle) {
      if (answers.lifestyle === 'Smoke' || answers.lifestyle === 'Both') {
        const smokingSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('cough') || s.toLowerCase().includes('breath') ||
          s.toLowerCase().includes('smoking')
        );
        smokingSymptoms.forEach(s => symptoms[s] = true);
        if (smokingSymptoms.length > 0) {
          influencingFactors.push('Smoking history - added respiratory concerns');
        }
      }
      if (answers.lifestyle === 'Drink' || answers.lifestyle === 'Both') {
        const alcoholSymptoms = availableSymptoms.filter(s =>
          s.toLowerCase().includes('alcohol') || s.toLowerCase().includes('liver')
        );
        alcoholSymptoms.forEach(s => symptoms[s] = true);
        if (alcoholSymptoms.length > 0) {
          influencingFactors.push('Alcohol consumption - considered related issues');
        }
      }
    }

    return { symptoms, influencingFactors };
  };

  const analyzeSymptomsAndPredict = async () => {
    setLoading(true);
    
    // Add analyzing message
    const analyzingMessage = {
      sender: 'ai',
      text: '🔍 Analyzing your symptoms...',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, analyzingMessage]);

    try {
      // Map answers to symptoms
      const { symptoms, influencingFactors } = mapAnswersToSymptoms(userAnswers);
      
      // Count active symptoms
      const activeSymptomCount = Object.values(symptoms).filter(Boolean).length;
      
      // Make prediction
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      
      // Add result message with context
      setTimeout(() => {
        const contextInfo = influencingFactors.length > 0 
          ? `\n\n📋 **Analysis Context:**\n${influencingFactors.map((f, i) => `${i + 1}. ${f}`).join('\n')}\n\n🔬 **Total symptoms analyzed:** ${activeSymptomCount} symptoms`
          : '';
        
        const resultMessage = {
          sender: 'ai',
          text: `Based on your symptoms and medical history, you might have:\n\n🏥 **${data.disease}**\n\n📊 Accuracy: ${data.confidence?.toFixed(1)}%${contextInfo}\n\n⚠️ **Important:** This is an AI prediction based on the information you provided. This should not replace professional medical advice. Please consult a healthcare provider for accurate diagnosis and treatment.`,
          timestamp: new Date(),
          isResult: true,
        };
        setMessages(prev => [...prev, resultMessage]);
        setLoading(false);
      }, 1500);
      
    } catch (err) {
      const errorMessage = {
        sender: 'ai',
        text: '❌ Sorry, I encountered an error. Please make sure the Flask API is running and try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      setLoading(false);
    }
  };

  const resetChat = () => {
    setMessages([]);
    setCurrentQuestion(0);
    setUserAnswers({});
    setUserInput('');
    setChatStarted(false);
    setLoading(false);
  };

  return (
    <div className="symptom-checker">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <div className="avatar">🏥</div>
            <div className="header-info">
              <h1>AI Health Assistant</h1>
              <p className="status">Online</p>
            </div>
          </div>
          {chatStarted && (
            <button className="reset-btn" onClick={resetChat}>
              🔄 Reset
            </button>
          )}
        </div>

        <div className="chat-messages">
          {!chatStarted ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🏥</div>
              <h2>Welcome to AI Health Assistant</h2>
              <p>I'll help you identify potential health conditions based on your symptoms.</p>
              <button className="start-chat-btn" onClick={startChat}>
                Start Consultation
              </button>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div key={index} className={`message ${message.sender}`}>
                  <div className="message-content">
                    <div className="message-bubble">
                      {message.text.split('\n').map((line, i) => (
                        <p key={i}>{line}</p>
                      ))}
                    </div>
                    <div className="message-time">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  
                  {message.type === 'multiple' && message.options && message.options.length > 0 && index === messages.length - 1 && !loading && (
                    <div className="symptom-selection">
                      <div className="search-box">
                        <input
                          type="text"
                          placeholder="🔍 Search symptoms (e.g., fever, headache, pain)..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="symptom-search"
                        />
                        {searchTerm && (
                          <button 
                            className="clear-search"
                            onClick={() => setSearchTerm('')}
                          >
                            ✕
                          </button>
                        )}
                      </div>
                      <div className="symptom-grid">
                        {filterSymptoms(message.options, searchTerm).length > 0 ? (
                          filterSymptoms(message.options, searchTerm).map((option, optIndex) => (
                            <label
                              key={optIndex}
                              className={`symptom-checkbox ${selectedSymptoms.includes(option) ? 'selected' : ''}`}
                            >
                              <input
                                type="checkbox"
                                checked={selectedSymptoms.includes(option)}
                                onChange={() => handleSymptomToggle(option)}
                              />
                              <span>{formatSymptomName(option)}</span>
                            </label>
                          ))
                        ) : (
                          <div className="no-results">
                            No symptoms found for "{searchTerm}"
                          </div>
                        )}
                      </div>
                      <div className="symptom-actions">
                        <div className="selected-count">
                          {selectedSymptoms.length} symptom{selectedSymptoms.length !== 1 ? 's' : ''} selected
                        </div>
                        <button
                          className="confirm-btn"
                          onClick={handleSymptomConfirm}
                          disabled={selectedSymptoms.length === 0}
                        >
                          Continue ➤
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {message.type === 'buttons' && message.options && index === messages.length - 1 && !loading && (
                    <div className="button-options">
                      {message.options.map((option, optIndex) => (
                        <button
                          key={optIndex}
                          className="option-btn"
                          onClick={() => handleUserResponse(option)}
                        >
                          {option}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SymptomChecker;
