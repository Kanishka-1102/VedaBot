"""
Ayurvedic Body Type (Dosha) Assessment Module
This module provides comprehensive dosha assessment functionality for VedaBot.
"""

import streamlit as st
from typing import Dict, List, Tuple
import json

class DoshaAssessment:
    """Class to handle Ayurvedic dosha assessment and body type prediction."""
    
    def __init__(self):
        self.questions = self._load_questions()
        self.dosha_characteristics = self._load_dosha_characteristics()
    
    def _load_questions(self) -> List[Dict]:
        """Load comprehensive dosha assessment questions."""
        return [
            # Physical Characteristics
            {
                "category": "Physical Build",
                "question": "What best describes your body frame?",
                "options": {
                    "Vata": "Thin, light, delicate frame",
                    "Pitta": "Medium build, muscular, well-proportioned",
                    "Kapha": "Large, solid, heavy frame"
                }
            },
            {
                "category": "Physical Build", 
                "question": "How would you describe your weight?",
                "options": {
                    "Vata": "Underweight or difficulty gaining weight",
                    "Pitta": "Normal weight, easy to maintain",
                    "Kapha": "Overweight or tendency to gain weight easily"
                }
            },
            {
                "category": "Physical Build",
                "question": "What about your height?",
                "options": {
                    "Vata": "Very tall or very short",
                    "Pitta": "Average height",
                    "Kapha": "Short to medium height"
                }
            },
            {
                "category": "Physical Build",
                "question": "How are your joints?",
                "options": {
                    "Vata": "Thin, prominent, cracking sounds",
                    "Pitta": "Medium-sized, well-formed",
                    "Kapha": "Large, well-padded, smooth"
                }
            },
            
            # Skin Characteristics
            {
                "category": "Skin",
                "question": "What best describes your skin?",
                "options": {
                    "Vata": "Dry, thin, cool, rough",
                    "Pitta": "Warm, sensitive, prone to rashes",
                    "Kapha": "Oily, thick, cool, smooth"
                }
            },
            {
                "category": "Skin",
                "question": "How does your skin react to sun?",
                "options": {
                    "Vata": "Burns easily, tans poorly",
                    "Pitta": "Burns easily, tans well",
                    "Kapha": "Tans easily, rarely burns"
                }
            },
            {
                "category": "Skin",
                "question": "What about moles and freckles?",
                "options": {
                    "Vata": "Few or none",
                    "Pitta": "Many moles and freckles",
                    "Kapha": "Some, but not many"
                }
            },
            
            # Hair Characteristics
            {
                "category": "Hair",
                "question": "What describes your hair best?",
                "options": {
                    "Vata": "Dry, thin, curly, brittle",
                    "Pitta": "Fine, straight, early graying",
                    "Kapha": "Thick, oily, wavy, lustrous"
                }
            },
            {
                "category": "Hair",
                "question": "How is your hair growth?",
                "options": {
                    "Vata": "Slow growth, prone to breakage",
                    "Pitta": "Normal growth, early graying",
                    "Kapha": "Fast growth, thick and strong"
                }
            },
            
            # Eyes
            {
                "category": "Eyes",
                "question": "What best describes your eyes?",
                "options": {
                    "Vata": "Small, dry, dark, restless",
                    "Pitta": "Medium, sharp, light-colored, penetrating",
                    "Kapha": "Large, attractive, thick lashes, calm"
                }
            },
            {
                "category": "Eyes",
                "question": "How do your eyes feel?",
                "options": {
                    "Vata": "Dry, tired, sensitive to light",
                    "Pitta": "Burning, bloodshot when stressed",
                    "Kapha": "Watery, heavy, droopy"
                }
            },
            
            # Appetite and Digestion
            {
                "category": "Digestion",
                "question": "How is your appetite?",
                "options": {
                    "Vata": "Irregular, sometimes forget to eat",
                    "Pitta": "Strong, regular, can't skip meals",
                    "Kapha": "Steady, can skip meals easily"
                }
            },
            {
                "category": "Digestion",
                "question": "What about your digestion?",
                "options": {
                    "Vata": "Irregular, gas, bloating",
                    "Pitta": "Strong, quick, sometimes loose stools",
                    "Kapha": "Slow, steady, heavy feeling after meals"
                }
            },
            {
                "category": "Digestion",
                "question": "How do you feel about spicy food?",
                "options": {
                    "Vata": "Can't handle much spice",
                    "Pitta": "Love spicy food, crave it",
                    "Kapha": "Moderate spice tolerance"
                }
            },
            
            # Energy and Activity
            {
                "category": "Energy",
                "question": "What describes your energy pattern?",
                "options": {
                    "Vata": "Bursts of energy, then tired",
                    "Pitta": "Steady, intense energy",
                    "Kapha": "Slow to start, steady endurance"
                }
            },
            {
                "category": "Energy",
                "question": "How do you handle physical activity?",
                "options": {
                    "Vata": "Love variety, get bored easily",
                    "Pitta": "Intense, competitive activities",
                    "Kapha": "Steady, routine activities"
                }
            },
            {
                "category": "Energy",
                "question": "What about your sleep pattern?",
                "options": {
                    "Vata": "Light sleeper, irregular hours",
                    "Pitta": "Moderate sleep, wake up hot",
                    "Kapha": "Deep sleeper, hard to wake up"
                }
            },
            
            # Mental Characteristics
            {
                "category": "Mental",
                "question": "How do you learn best?",
                "options": {
                    "Vata": "Quick to learn, quick to forget",
                    "Pitta": "Sharp, focused, analytical",
                    "Kapha": "Slow to learn, excellent memory"
                }
            },
            {
                "category": "Mental",
                "question": "What about your decision making?",
                "options": {
                    "Vata": "Quick decisions, change mind often",
                    "Pitta": "Decisive, logical, sometimes impulsive",
                    "Kapha": "Slow, deliberate, stick to decisions"
                }
            },
            {
                "category": "Mental",
                "question": "How do you handle stress?",
                "options": {
                    "Vata": "Worry, anxiety, scattered thoughts",
                    "Pitta": "Anger, irritability, criticism",
                    "Kapha": "Withdrawal, denial, avoidance"
                }
            },
            
            # Emotional Characteristics
            {
                "category": "Emotional",
                "question": "What's your typical mood?",
                "options": {
                    "Vata": "Enthusiastic, anxious, changeable",
                    "Pitta": "Intense, passionate, irritable",
                    "Kapha": "Calm, content, sometimes lethargic"
                }
            },
            {
                "category": "Emotional",
                "question": "How do you express emotions?",
                "options": {
                    "Vata": "Quick to laugh or cry, expressive",
                    "Pitta": "Intense, direct, sometimes harsh",
                    "Kapha": "Steady, slow to anger, nurturing"
                }
            },
            
            # Weather Preferences
            {
                "category": "Weather",
                "question": "What weather do you prefer?",
                "options": {
                    "Vata": "Warm, humid weather",
                    "Pitta": "Cool, dry weather",
                    "Kapha": "Warm, dry weather"
                }
            },
            {
                "category": "Weather",
                "question": "How do you handle cold weather?",
                "options": {
                    "Vata": "Very sensitive to cold",
                    "Pitta": "Moderate tolerance",
                    "Kapha": "Good tolerance, prefer it"
                }
            },
            
            # Additional Characteristics
            {
                "category": "General",
                "question": "What about your voice?",
                "options": {
                    "Vata": "High-pitched, fast, talkative",
                    "Pitta": "Sharp, clear, commanding",
                    "Kapha": "Deep, slow, melodious"
                }
            },
            {
                "category": "General",
                "question": "How do you walk?",
                "options": {
                    "Vata": "Fast, light, irregular pace",
                    "Pitta": "Purposeful, determined stride",
                    "Kapha": "Slow, steady, graceful"
                }
            }
        ]
    
    def _load_dosha_characteristics(self) -> Dict:
        """Load detailed dosha characteristics and recommendations."""
        return {
            "Vata": {
                "name": "Vata Dosha",
                "elements": "Air + Space",
                "qualities": "Cold, dry, light, mobile, rough, subtle",
                "description": "Vata governs movement, circulation, breathing, and elimination. People with dominant Vata are creative, energetic, and quick-thinking.",
                "physical_traits": [
                    "Thin, light build",
                    "Dry skin and hair",
                    "Cold hands and feet",
                    "Irregular appetite and digestion",
                    "Light, interrupted sleep"
                ],
                "mental_traits": [
                    "Quick to learn and forget",
                    "Creative and artistic",
                    "Enthusiastic and energetic",
                    "Anxious when out of balance",
                    "Tendency to worry"
                ],
                "recommendations": {
                    "diet": [
                        "Warm, cooked foods",
                        "Sweet, sour, and salty tastes",
                        "Regular meal times",
                        "Avoid cold, raw foods",
                        "Include healthy fats like ghee and olive oil"
                    ],
                    "lifestyle": [
                        "Regular routine and schedule",
                        "Gentle, grounding exercises like yoga",
                        "Warm oil massage (abhyanga)",
                        "Adequate rest and sleep",
                        "Avoid excessive travel and stimulation"
                    ],
                    "herbs": [
                        "Ashwagandha (for grounding)",
                        "Brahmi (for mental calm)",
                        "Shatavari (for nourishment)",
                        "Triphala (for digestion)",
                        "Ginger (for warming)"
                    ]
                }
            },
            "Pitta": {
                "name": "Pitta Dosha", 
                "elements": "Fire + Water",
                "qualities": "Hot, sharp, light, oily, liquid, mobile",
                "description": "Pitta governs digestion, metabolism, and transformation. People with dominant Pitta are intelligent, focused, and natural leaders.",
                "physical_traits": [
                    "Medium build, muscular",
                    "Warm skin, prone to rashes",
                    "Strong appetite and digestion",
                    "Good circulation",
                    "Tendency to overheat"
                ],
                "mental_traits": [
                    "Sharp intellect and memory",
                    "Focused and determined",
                    "Natural leadership qualities",
                    "Perfectionist tendencies",
                    "Can be critical when imbalanced"
                ],
                "recommendations": {
                    "diet": [
                        "Cooling foods and drinks",
                        "Sweet, bitter, and astringent tastes",
                        "Avoid spicy, sour, salty foods",
                        "Fresh fruits and vegetables",
                        "Moderate protein intake"
                    ],
                    "lifestyle": [
                        "Cool, calm environment",
                        "Regular exercise, avoid overheating",
                        "Meditation and relaxation",
                        "Avoid excessive competition",
                        "Regular meal times"
                    ],
                    "herbs": [
                        "Amla (for cooling)",
                        "Neem (for purification)",
                        "Brahmi (for mental calm)",
                        "Shatavari (for cooling)",
                        "Coriander (for digestion)"
                    ]
                }
            },
            "Kapha": {
                "name": "Kapha Dosha",
                "elements": "Earth + Water", 
                "qualities": "Heavy, slow, cool, oily, smooth, dense",
                "description": "Kapha governs structure, stability, and lubrication. People with dominant Kapha are calm, loving, and have great endurance.",
                "physical_traits": [
                    "Large, solid build",
                    "Oily, smooth skin",
                    "Thick, lustrous hair",
                    "Strong bones and joints",
                    "Slow metabolism"
                ],
                "mental_traits": [
                    "Calm and steady",
                    "Excellent memory",
                    "Loving and nurturing",
                    "Slow to anger",
                    "Tendency toward complacency"
                ],
                "recommendations": {
                    "diet": [
                        "Light, warm foods",
                        "Pungent, bitter, astringent tastes",
                        "Avoid heavy, oily foods",
                        "Eat smaller portions",
                        "Include warming spices"
                    ],
                    "lifestyle": [
                        "Regular vigorous exercise",
                        "Variety and stimulation",
                        "Early morning routine",
                        "Avoid excessive sleep",
                        "Stay active and engaged"
                    ],
                    "herbs": [
                        "Ginger (for stimulation)",
                        "Turmeric (for purification)",
                        "Triphala (for digestion)",
                        "Brahmi (for mental clarity)",
                        "Tulsi (for energy)"
                    ]
                }
            }
        }
    
    def calculate_dosha_scores(self, answers: Dict[str, str]) -> Dict[str, int]:
        """Calculate dosha scores based on user answers."""
        scores = {"Vata": 0, "Pitta": 0, "Kapha": 0}
        
        for question_id, answer in answers.items():
            if answer in scores:
                scores[answer] += 1
        
        return scores
    
    def determine_primary_dosha(self, scores: Dict[str, int]) -> Tuple[str, Dict[str, int]]:
        """Determine the primary dosha based on scores."""
        primary_dosha = max(scores, key=scores.get)
        total_score = sum(scores.values())
        
        # Calculate percentages
        percentages = {
            dosha: round((score / total_score) * 100, 1) if total_score > 0 else 0
            for dosha, score in scores.items()
        }
        
        return primary_dosha, percentages
    
    def get_dosha_analysis(self, primary_dosha: str, percentages: Dict[str, int]) -> str:
        """Generate detailed dosha analysis and recommendations."""
        dosha_info = self.dosha_characteristics[primary_dosha]
        
        # Determine secondary dosha
        sorted_doshas = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        secondary_dosha = sorted_doshas[1][0] if len(sorted_doshas) > 1 else None
        
        analysis = f"""
        ## 🧘‍♀️ Your Ayurvedic Body Type Assessment
        
        ### Primary Dosha: {dosha_info['name']} ({percentages[primary_dosha]}%)
        **Elements:** {dosha_info['elements']}  
        **Qualities:** {dosha_info['qualities']}
        
        {dosha_info['description']}
        
        ### 📊 Your Dosha Breakdown:
        - **{primary_dosha}:** {percentages[primary_dosha]}%
        - **{sorted_doshas[1][0] if len(sorted_doshas) > 1 else 'N/A'}:** {sorted_doshas[1][1] if len(sorted_doshas) > 1 else 0}%
        - **{sorted_doshas[2][0] if len(sorted_doshas) > 2 else 'N/A'}:** {sorted_doshas[2][1] if len(sorted_doshas) > 2 else 0}%
        
        ### 🎯 Physical Characteristics:
        """
        
        for trait in dosha_info['physical_traits']:
            analysis += f"- {trait}\n"
        
        analysis += "\n### 🧠 Mental Characteristics:\n"
        for trait in dosha_info['mental_traits']:
            analysis += f"- {trait}\n"
        
        analysis += f"\n### 🍽️ Dietary Recommendations for {primary_dosha}:\n"
        for rec in dosha_info['recommendations']['diet']:
            analysis += f"- {rec}\n"
        
        analysis += f"\n### 🏃‍♀️ Lifestyle Recommendations for {primary_dosha}:\n"
        for rec in dosha_info['recommendations']['lifestyle']:
            analysis += f"- {rec}\n"
        
        analysis += f"\n### 🌿 Beneficial Herbs for {primary_dosha}:\n"
        for herb in dosha_info['recommendations']['herbs']:
            analysis += f"- {herb}\n"
        
        if secondary_dosha and percentages[secondary_dosha] > 20:
            secondary_info = self.dosha_characteristics[secondary_dosha]
            analysis += f"\n### ⚖️ Secondary Dosha Influence: {secondary_dosha}\n"
            analysis += f"Your {secondary_dosha} influence ({percentages[secondary_dosha]}%) means you may also benefit from:\n"
            for rec in secondary_info['recommendations']['diet'][:3]:  # Top 3 recommendations
                analysis += f"- {rec}\n"
        
        analysis += f"""
        
        ### 💡 Key Insights:
        - **Balance your {primary_dosha}** by following the recommendations above
        - **Listen to your body** and adjust based on seasonal changes
        - **Consult an Ayurvedic practitioner** for personalized guidance
        - **Remember:** This assessment is a starting point - individual variations exist
        
        ---
        **Note:** This assessment is for educational purposes. For personalized Ayurvedic guidance, consult a qualified practitioner.
        """
        
        return analysis
    
    def run_assessment(self) -> Tuple[str, Dict[str, int]]:
        """Run the complete dosha assessment."""
        st.markdown("## 🧘‍♀️ Discover Your Ayurvedic Body Type")
        st.markdown("Answer these questions to discover your dominant dosha (Vata, Pitta, or Kapha) and get personalized recommendations.")
        
        answers = {}
        
        # Group questions by category
        categories = {}
        for i, question in enumerate(self.questions):
            category = question['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((i, question))
        
        # Display questions by category
        for category, questions in categories.items():
            st.markdown(f"### {category}")
            
            for question_id, question in questions:
                answer = st.radio(
                    question['question'],
                    options=list(question['options'].keys()),
                    format_func=lambda x: question['options'][x],
                    key=f"question_{question_id}",
                    horizontal=False
                )
                answers[f"question_{question_id}"] = answer
            
            st.markdown("---")
        
        if st.button("🔮 Discover My Body Type", type="primary"):
            if len(answers) == len(self.questions):
                # Calculate scores
                scores = self.calculate_dosha_scores(answers)
                primary_dosha, percentages = self.determine_primary_dosha(scores)
                
                # Generate analysis
                analysis = self.get_dosha_analysis(primary_dosha, percentages)
                
                # Display results
                st.markdown(analysis)
                
                # Save results to session state
                st.session_state['dosha_results'] = {
                    'primary_dosha': primary_dosha,
                    'percentages': percentages,
                    'scores': scores,
                    'analysis': analysis
                }
                
                return analysis, percentages
            else:
                st.error("Please answer all questions to get your body type assessment.")
                return None, None
        
        return None, None

def get_dosha_icon(dosha: str) -> str:
    """Get appropriate emoji/icon for each dosha."""
    icons = {
        "Vata": "🌪️",  # Wind/Air
        "Pitta": "🔥",  # Fire
        "Kapha": "🌊"   # Water/Earth
    }
    return icons.get(dosha, "🧘‍♀️")

def get_dosha_color(dosha: str) -> str:
    """Get color theme for each dosha."""
    colors = {
        "Vata": "#E3F2FD",  # Light blue
        "Pitta": "#FFF3E0",  # Light orange
        "Kapha": "#E8F5E8"   # Light green
    }
    return colors.get(dosha, "#F5F5F5")
