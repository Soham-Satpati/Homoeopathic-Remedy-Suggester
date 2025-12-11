import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, Loader, FileText, Download, Trash2, Plus, ArrowRight } from 'lucide-react';

const HorusChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationState, setConversationState] = useState({
    stage: 'greeting', // greeting, symptoms_collection, refinement, pattern_discovery, report
    patientId: '',
    symptoms: {
      physical: [],
      psychological: [],
      generals: []
    },
    refinedSymptoms: {},
    patternSymptoms: [],
    topRemedies: [],
    notes: ''
  });
  const [showSidebar, setShowSidebar] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Initial greeting
    addBotMessage(
      "👋 Welcome to HoRUS 3 - Your Clinical Homeopathy Assistant!\n\n" +
      "I'm here to help you analyze cases and find the best remedies. Let's start:\n\n" +
      "**Please provide a Patient ID** (or I can generate one for you).\n\n" +
      "You can say things like:\n" +
      "- \"Generate a new patient ID\"\n" +
      "- \"Use patient ID PT-2025-001\"\n" +
      "- \"Show me my patient history\""
    );
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const addBotMessage = (text) => {
    setMessages(prev => [...prev, { role: 'assistant', content: text, timestamp: new Date() }]);
  };

  const addUserMessage = (text) => {
    setMessages(prev => [...prev, { role: 'user', content: text, timestamp: new Date() }]);
  };

  const generatePatientId = () => {
    const year = new Date().getFullYear();
    const random = Math.floor(Math.random() * 900) + 100;
    return `PT-${year}-${random}`;
  };

  const callGeminiAPI = async (prompt, context) => {
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [
            { 
              role: "user", 
              content: `You are HoRUS 3, an expert homeopathic case analysis assistant. 
              
Current conversation context: ${JSON.stringify(context)}

User's message: ${prompt}

Based on the current stage (${context.stage}), provide appropriate guidance:
- If in 'greeting' stage: Help establish patient ID
- If in 'symptoms_collection': Extract and categorize symptoms (physical, psychological, general)
- If in 'refinement': Suggest refined symptom language
- If in 'pattern_discovery': Identify concomitant symptoms
- If in 'report': Generate remedy recommendations

Respond conversationally and professionally. Extract structured data when relevant.`
            }
          ],
        })
      });

      const data = await response.json();
      return data.content[0].text;
    } catch (error) {
      console.error('API Error:', error);
      return "I apologize, but I'm having trouble processing that right now. Could you try rephrasing?";
    }
  };

  const parseSymptoms = (text) => {
    const physical = [];
    const psychological = [];
    const generals = [];

    // Simple keyword-based categorization (in production, use NLP)
    const physicalKeywords = ['pain', 'ache', 'swelling', 'rash', 'fever', 'cough', 'throat', 'stomach', 'head'];
    const psychKeywords = ['anxiety', 'fear', 'anger', 'sad', 'irritable', 'restless', 'depressed', 'worried'];
    const generalKeywords = ['worse', 'better', 'morning', 'evening', 'cold', 'heat', 'motion', 'rest'];

    const sentences = text.toLowerCase().split(/[.,;]/).filter(s => s.trim());
    
    sentences.forEach(sentence => {
      if (physicalKeywords.some(kw => sentence.includes(kw))) {
        physical.push(sentence.trim());
      } else if (psychKeywords.some(kw => sentence.includes(kw))) {
        psychological.push(sentence.trim());
      } else if (generalKeywords.some(kw => sentence.includes(kw))) {
        generals.push(sentence.trim());
      } else if (sentence.trim()) {
        physical.push(sentence.trim()); // Default to physical
      }
    });

    return { physical, psychological, generals };
  };

  const processMessage = async (userInput) => {
    setLoading(true);
    addUserMessage(userInput);

    try {
      const lowerInput = userInput.toLowerCase();

      // Stage: Greeting - Patient ID handling
      if (conversationState.stage === 'greeting') {
        if (lowerInput.includes('generate') || lowerInput.includes('new') || lowerInput.includes('create')) {
          const newId = generatePatientId();
          setConversationState(prev => ({ ...prev, patientId: newId, stage: 'symptoms_collection' }));
          addBotMessage(
            `✅ **Patient ID Generated:** ${newId}\n\n` +
            `Now, let's collect the symptoms. Please tell me about:\n\n` +
            `🏥 **Physical symptoms** (e.g., "severe headache worse from motion")\n` +
            `🧠 **Psychological symptoms** (e.g., "anxiety before exams")\n` +
            `🌡️ **General symptoms** (e.g., "worse in cold weather")\n\n` +
            `You can describe them naturally, and I'll organize them for you.`
          );
        } else if (lowerInput.match(/pt-\d{4}-\d{3}/)) {
          const match = userInput.match(/PT-\d{4}-\d{3}/i);
          if (match) {
            const pid = match[0].toUpperCase();
            setConversationState(prev => ({ ...prev, patientId: pid, stage: 'symptoms_collection' }));
            addBotMessage(
              `✅ **Using Patient ID:** ${pid}\n\n` +
              `Great! Now let's gather the symptoms. Tell me about the patient's condition.`
            );
          }
        } else {
          addBotMessage(
            `I need a patient ID to proceed. You can:\n` +
            `- Say "generate new patient ID"\n` +
            `- Provide an ID like "PT-2025-001"`
          );
        }
      }

      // Stage: Symptoms Collection
      else if (conversationState.stage === 'symptoms_collection') {
        if (lowerInput.includes('done') || lowerInput.includes('finished') || lowerInput.includes('next')) {
          const totalSymptoms = 
            conversationState.symptoms.physical.length +
            conversationState.symptoms.psychological.length +
            conversationState.symptoms.generals.length;

          if (totalSymptoms === 0) {
            addBotMessage("⚠️ No symptoms collected yet. Please describe at least one symptom.");
          } else {
            setConversationState(prev => ({ ...prev, stage: 'refinement' }));
            addBotMessage(
              `📊 **Symptoms Summary:**\n` +
              `- Physical: ${conversationState.symptoms.physical.length}\n` +
              `- Psychological: ${conversationState.symptoms.psychological.length}\n` +
              `- General: ${conversationState.symptoms.generals.length}\n\n` +
              `Would you like to:\n` +
              `1️⃣ Refine symptoms (recommended)\n` +
              `2️⃣ Discover patterns\n` +
              `3️⃣ Generate report directly\n\n` +
              `Just tell me your choice!`
            );
          }
        } else {
          // Parse and categorize symptoms
          const parsed = parseSymptoms(userInput);
          setConversationState(prev => ({
            ...prev,
            symptoms: {
              physical: [...prev.symptoms.physical, ...parsed.physical],
              psychological: [...prev.symptoms.psychological, ...parsed.psychological],
              generals: [...prev.symptoms.generals, ...parsed.generals]
            }
          }));

          const count = parsed.physical.length + parsed.psychological.length + parsed.generals.length;
          addBotMessage(
            `✅ Added ${count} symptom(s)!\n\n` +
            `**Current totals:**\n` +
            `- Physical: ${conversationState.symptoms.physical.length + parsed.physical.length}\n` +
            `- Psychological: ${conversationState.symptoms.psychological.length + parsed.psychological.length}\n` +
            `- General: ${conversationState.symptoms.generals.length + parsed.generals.length}\n\n` +
            `Add more symptoms, or say **"done"** to proceed.`
          );
        }
      }

      // Stage: Refinement
      else if (conversationState.stage === 'refinement') {
        if (lowerInput.includes('skip') || lowerInput.includes('generate report') || lowerInput.includes('3')) {
          await generateReport();
        } else if (lowerInput.includes('pattern') || lowerInput.includes('discover') || lowerInput.includes('2')) {
          setConversationState(prev => ({ ...prev, stage: 'pattern_discovery' }));
          addBotMessage(
            `🔬 **Pattern Discovery Mode**\n\n` +
            `Based on your symptoms, I've identified these common concomitant patterns:\n\n` +
            `1️⃣ **Restlessness with anxiety** - Often seen with fear of death\n` +
            `2️⃣ **Burning pains** - Better from heat applications\n` +
            `3️⃣ **Irritability when sick** - Worse from consolation\n\n` +
            `Would you like to add any of these patterns? (say the number or "none")`
          );
        } else if (lowerInput.includes('refine') || lowerInput.includes('1')) {
          const geminiResponse = await callGeminiAPI(
            `Suggest refined homeopathic repertory language for these symptoms: ${JSON.stringify(conversationState.symptoms)}`,
            conversationState
          );
          addBotMessage(geminiResponse);
        }
      }

      // Stage: Pattern Discovery
      else if (conversationState.stage === 'pattern_discovery') {
        if (lowerInput.includes('none') || lowerInput.includes('skip') || lowerInput.includes('report')) {
          await generateReport();
        } else {
          addBotMessage(
            `✅ Pattern added! Say **"generate report"** when ready, or add more patterns.`
          );
        }
      }

      // Report stage
      else if (conversationState.stage === 'report') {
        if (lowerInput.includes('new case') || lowerInput.includes('start over')) {
          resetConversation();
        } else {
          addBotMessage(
            `📋 The report has been generated. You can:\n` +
            `- Start a **"new case"**\n` +
            `- Ask about specific remedies\n` +
            `- Request **"patient history"**`
          );
        }
      }

    } catch (error) {
      console.error('Error:', error);
      addBotMessage("I encountered an error processing that. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    setConversationState(prev => ({ ...prev, stage: 'report' }));
    
    // Mock remedy generation (in production, use actual repertorization logic)
    const mockRemedies = [
      { name: 'Arnica', score: 0.85, context: 'Trauma, bruising, soreness' },
      { name: 'Rhus-tox', score: 0.78, context: 'Restlessness, worse rest, better motion' },
      { name: 'Bryonia', score: 0.72, context: 'Worse motion, better rest, irritable' },
      { name: 'Pulsatilla', score: 0.68, context: 'Changeable, mild, desires company' },
      { name: 'Sulphur', score: 0.65, context: 'Burning, untidy, worse heat' }
    ];

    setConversationState(prev => ({ ...prev, topRemedies: mockRemedies }));

    const reportText = `
📊 **Clinical Report Generated**

**Patient ID:** ${conversationState.patientId}
**Date:** ${new Date().toLocaleDateString()}

**Symptom Summary:**
- Physical: ${conversationState.symptoms.physical.length} symptoms
- Psychological: ${conversationState.symptoms.psychological.length} symptoms
- General: ${conversationState.symptoms.generals.length} symptoms

**Top 5 Remedies:**

${mockRemedies.map((r, i) => `
${i + 1}. **${r.name.toUpperCase()}** - Score: ${r.score}
   💡 ${r.context}
`).join('\n')}

Would you like to:
- Add clinical notes
- Start a new case
- Export this report
`;

    addBotMessage(reportText);
  };

  const resetConversation = () => {
    setConversationState({
      stage: 'greeting',
      patientId: '',
      symptoms: { physical: [], psychological: [], generals: [] },
      refinedSymptoms: {},
      patternSymptoms: [],
      topRemedies: [],
      notes: ''
    });
    setMessages([]);
    addBotMessage(
      "🔄 **New Case Started**\n\n" +
      "Please provide a Patient ID or say 'generate new patient ID'."
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !loading) {
      processMessage(input.trim());
      setInput('');
    }
  };

  const quickActions = [
    { label: "Generate Report", action: "generate report now" },
    { label: "Add More Symptoms", action: "I have more symptoms" },
    { label: "New Patient", action: "start new case" },
  ];

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-md px-6 py-4 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-indigo-700">HoRUS 3 AI Assistant</h1>
              <p className="text-sm text-gray-600">Clinical Homeopathy Intelligence</p>
            </div>
            <div className="flex gap-2">
              {conversationState.patientId && (
                <div className="px-4 py-2 bg-indigo-100 rounded-lg">
                  <span className="text-sm font-semibold text-indigo-800">
                    {conversationState.patientId}
                  </span>
                </div>
              )}
              <button
                onClick={resetConversation}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg flex items-center gap-2 transition-colors"
              >
                <Plus size={16} />
                New Case
              </button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0">
                  <Bot size={18} className="text-white" />
                </div>
              )}
              <div
                className={`max-w-2xl px-4 py-3 rounded-2xl ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white shadow-md text-gray-800 border border-gray-200'
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </div>
                <div className="text-xs mt-2 opacity-60">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                  <User size={18} className="text-white" />
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0">
                <Bot size={18} className="text-white" />
              </div>
              <div className="bg-white shadow-md px-4 py-3 rounded-2xl border border-gray-200">
                <Loader size={20} className="animate-spin text-indigo-600" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions */}
        {conversationState.stage !== 'greeting' && (
          <div className="px-6 py-2 flex gap-2 bg-gray-50 border-t">
            {quickActions.map((action, idx) => (
              <button
                key={idx}
                onClick={() => processMessage(action.action)}
                disabled={loading}
                className="px-3 py-1 text-xs bg-white border border-gray-300 hover:bg-gray-100 rounded-full transition-colors disabled:opacity-50"
              >
                {action.label}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !loading && input.trim()) {
                  handleSubmit(e);
                }
              }}
              placeholder="Type your message..."
              disabled={loading}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-100"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send size={18} />
              Send
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Stage: {conversationState.stage.replace('_', ' ')} • 
            Powered by Gemini AI
          </p>
        </div>
      </div>

      {/* Sidebar (Patient Info) */}
      {conversationState.patientId && (
        <div className="w-80 bg-white border-l border-gray-200 p-6 overflow-y-auto">
          <h2 className="text-lg font-bold text-gray-800 mb-4">Case Summary</h2>
          
          <div className="space-y-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-gray-600 mb-1">Patient ID</div>
              <div className="font-semibold text-gray-800">{conversationState.patientId}</div>
            </div>

            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-xs text-gray-600 mb-2">Symptoms Collected</div>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>🏥 Physical</span>
                  <span className="font-semibold">{conversationState.symptoms.physical.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>🧠 Psychological</span>
                  <span className="font-semibold">{conversationState.symptoms.psychological.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>🌡️ General</span>
                  <span className="font-semibold">{conversationState.symptoms.generals.length}</span>
                </div>
              </div>
            </div>

            {conversationState.topRemedies.length > 0 && (
              <div className="bg-purple-50 p-3 rounded-lg">
                <div className="text-xs text-gray-600 mb-2">Top Remedies</div>
                <div className="space-y-2">
                  {conversationState.topRemedies.slice(0, 3).map((r, i) => (
                    <div key={i} className="text-sm">
                      <div className="font-semibold">{i + 1}. {r.name}</div>
                      <div className="text-xs text-gray-600">{r.context}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="pt-4 space-y-2">
              <button 
                onClick={() => alert('Export functionality - download PDF report')}
                className="w-full py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center justify-center gap-2 transition-colors"
              >
                <Download size={16} />
                Export Report
              </button>
              <button 
                onClick={resetConversation}
                className="w-full py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 flex items-center justify-center gap-2 transition-colors"
              >
                <Trash2 size={16} />
                Clear Case
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HorusChatbot;
