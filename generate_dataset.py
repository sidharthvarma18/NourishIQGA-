import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N = 2000

# --- Demographic Variables ---
age_groups = np.random.choice(
    ["18-24 (Student)", "25-34 (Young Adult)", "35-44 (Working Pro)", "45-54 (Middle-Aged)", "55+ (Senior)"],
    N, p=[0.28, 0.30, 0.22, 0.13, 0.07]
)
gender = np.random.choice(["Male", "Female", "Other"], N, p=[0.48, 0.50, 0.02])
city_tier = np.random.choice(["Metro", "Tier 2", "Tier 3", "Rural"], N, p=[0.30, 0.30, 0.25, 0.15])
occupation = np.where(
    age_groups == "18-24 (Student)", "Student",
    np.random.choice(["Working Professional", "Homemaker", "Self-Employed", "Retired", "Student"], N,
                     p=[0.38, 0.18, 0.18, 0.06, 0.20])
)
income_map = {
    "18-24 (Student)": np.random.choice(["Below ₹15k", "₹15k-₹30k"], N),
    "25-34 (Young Adult)": np.random.choice(["₹15k-₹30k", "₹30k-₹60k", "₹60k-₹1L"], N, p=[0.3, 0.4, 0.3]),
    "35-44 (Working Pro)": np.random.choice(["₹30k-₹60k", "₹60k-₹1L", "Above ₹1L"], N, p=[0.3, 0.4, 0.3]),
    "45-54 (Middle-Aged)": np.random.choice(["₹30k-₹60k", "₹60k-₹1L", "Above ₹1L"], N, p=[0.35, 0.35, 0.3]),
    "55+ (Senior)": np.random.choice(["₹15k-₹30k", "₹30k-₹60k", "₹60k-₹1L"], N, p=[0.3, 0.4, 0.3])
}
income_band = np.array([income_map[ag][i] for i, ag in enumerate(age_groups)])
education = np.random.choice(
    ["Below 12th", "12th Pass", "Graduate", "Post-Graduate", "Doctorate"],
    N, p=[0.05, 0.12, 0.42, 0.36, 0.05]
)
household_size = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.07, 0.13, 0.25, 0.30, 0.18, 0.07])

# --- Physical & Health ---
height = np.round(np.random.normal(165, 10, N), 1).clip(140, 195)
weight = np.round(np.random.normal(70, 14, N), 1).clip(38, 130)
bmi = np.round(weight / ((height / 100) ** 2), 1)
bmi_category = np.where(bmi < 18.5, "Underweight",
               np.where(bmi < 25, "Normal",
               np.where(bmi < 30, "Overweight", "Obese")))
health_conditions = np.random.choice(
    ["None", "Diabetes", "Hypertension", "Thyroid", "PCOS", "Heart Condition", "Anaemia"],
    N, p=[0.52, 0.15, 0.14, 0.09, 0.05, 0.03, 0.02]
)
food_allergies = np.random.choice(["None", "Lactose", "Gluten", "Nuts", "Shellfish"], N, p=[0.62, 0.16, 0.10, 0.08, 0.04])
medical_restrictions = np.random.choice(["None", "Low Salt", "Low Sugar", "Low Fat", "High Protein"], N, p=[0.55, 0.18, 0.14, 0.08, 0.05])
doctor_recommendation = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
family_health_history = np.random.choice(["Diabetes", "Heart Disease", "Obesity", "None", "Hypertension"], N, p=[0.25, 0.15, 0.20, 0.25, 0.15])
health_concern_level = np.random.choice(["Low", "Medium", "High"], N, p=[0.22, 0.42, 0.36])

# --- Diet and Food Habits ---
food_preference = np.random.choice(["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian"], N, p=[0.40, 0.42, 0.08, 0.10])
meals_per_day = np.random.choice([2, 3, 4, 5], N, p=[0.15, 0.50, 0.27, 0.08])
eating_outside_freq = np.random.choice(["Rarely", "1-2 times/week", "3-4 times/week", "Daily"], N, p=[0.22, 0.38, 0.28, 0.12])
eating_habit_quality = np.random.choice(["Poor", "Average", "Good", "Excellent"], N, p=[0.20, 0.38, 0.30, 0.12])
preferred_cuisine = np.random.choice(["North Indian", "South Indian", "Continental", "Chinese", "Mix"], N, p=[0.32, 0.28, 0.12, 0.10, 0.18])
meal_timing_regularity = np.random.choice(["Irregular", "Somewhat Regular", "Regular"], N, p=[0.30, 0.38, 0.32])
junk_food_freq = np.random.choice(["Rarely", "1-2x/week", "3-4x/week", "Daily"], N, p=[0.22, 0.38, 0.28, 0.12])
sugar_intake = np.random.choice(["Low", "Medium", "High"], N, p=[0.28, 0.44, 0.28])
water_intake = np.random.choice(["<1L", "1-2L", "2-3L", ">3L"], N, p=[0.18, 0.38, 0.32, 0.12])
diet_break_response = np.random.choice(["Give up", "Restart next day", "Restart next week", "Stay consistent"], N, p=[0.18, 0.34, 0.28, 0.20])

# --- Fitness & Lifestyle ---
activity_level = np.random.choice(["Sedentary", "Lightly Active", "Moderately Active", "Highly Active"], N, p=[0.28, 0.32, 0.28, 0.12])
fitness_goal = np.random.choice(
    ["Weight Loss", "Muscle Gain", "Energy Improvement", "General Wellness", "Condition Management"],
    N, p=[0.35, 0.22, 0.18, 0.17, 0.08]
)
workout_preference = np.random.choice(["Home Workout", "Gym", "Yoga", "Outdoor", "No Preference"], N, p=[0.32, 0.28, 0.18, 0.12, 0.10])
sleep_hours = np.random.choice(["<5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", ">8 hrs"], N, p=[0.08, 0.18, 0.32, 0.30, 0.12])
health_motivation = np.random.choice(["Low", "Medium", "High"], N, p=[0.20, 0.40, 0.40])
exercise_consistency = np.random.choice(["Never", "Occasionally", "2-3x/week", "5+x/week"], N, p=[0.22, 0.32, 0.30, 0.16])
workout_location = np.random.choice(["Home", "Gym", "Park", "No Fixed Place"], N, p=[0.38, 0.28, 0.18, 0.16])
motivation_dropoff = np.random.choice(["Lack of Time", "No Results", "Boredom", "Cost", "No Coach"], N, p=[0.30, 0.25, 0.18, 0.15, 0.12])
lifestyle_risk = np.random.choice(["Low", "Medium", "High"], N, p=[0.28, 0.42, 0.30])

# --- App Behaviour & Psychographics ---
health_personality = np.random.choice(["Proactive", "Reactive", "Passive"], N, p=[0.32, 0.38, 0.30])
data_sharing_comfort = np.random.choice(["Not Comfortable", "Neutral", "Comfortable"], N, p=[0.22, 0.38, 0.40])
health_influencer = np.random.choice(["Social Media", "Doctor", "Family/Friends", "Fitness Apps", "Self"], N, p=[0.28, 0.22, 0.25, 0.15, 0.10])
app_stop_reason = np.random.choice(["Too Expensive", "Not Useful", "Privacy Concern", "Complexity", "No Reason"], N, p=[0.28, 0.22, 0.18, 0.15, 0.17])
trust_app_reco = np.random.choice(["Low", "Medium", "High"], N, p=[0.22, 0.40, 0.38])
trust_ai_diet = np.random.choice(["Low", "Medium", "High"], N, p=[0.25, 0.38, 0.37])
reminder_preference = np.random.choice(["Push Notification", "SMS", "WhatsApp", "Email", "None"], N, p=[0.35, 0.22, 0.25, 0.12, 0.06])
barrier_to_app = np.random.choice(["Too Expensive", "Privacy", "Not Tech-Savvy", "No Time", "No Need"], N, p=[0.28, 0.20, 0.18, 0.20, 0.14])
notif_channel = np.random.choice(["WhatsApp", "App Notification", "SMS", "Email"], N, p=[0.38, 0.32, 0.18, 0.12])
past_health_app = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])

# --- Product & Service Interest ---
services_wanted = np.random.choice(
    ["BMI Tracker", "Diet Plans", "Workout Routines", "Hydration Reminders", "Progress Reports",
     "Recipe Suggestions", "Expert Chat", "Travel Diet", "AI Form Correction", "All Services"],
    N, p=[0.10, 0.18, 0.14, 0.10, 0.12, 0.10, 0.08, 0.06, 0.06, 0.06]
)
meal_plan_type = np.random.choice(
    ["Weight Loss", "Diabetic-Friendly", "High Protein", "Vegetarian Wellness", "Family Plan", "General Balanced"],
    N, p=[0.30, 0.18, 0.18, 0.14, 0.10, 0.10]
)
paid_health_before = np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
upgrade_trigger = np.random.choice(["Free Trial", "Doctor Advice", "Friends Referral", "Social Media Ad", "Discount"], N, p=[0.28, 0.22, 0.20, 0.18, 0.12])
interest_expert_chat = np.random.choice(["Yes", "No"], N, p=[0.55, 0.45])
interest_recipe_lib = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
interest_travel_diet = np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
interest_ai_form = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
interest_ad_free = np.random.choice(["Yes", "No"], N, p=[0.60, 0.40])
interest_personalised = np.random.choice(["Yes", "No"], N, p=[0.72, 0.28])

# --- Spending & Purchase ---
past_health_spend = np.random.choice(["<₹500", "₹500-₹1000", "₹1000-₹2000", ">₹2000"], N, p=[0.28, 0.30, 0.25, 0.17])
pricing_model = np.random.choice(["Monthly", "Quarterly", "Annual", "Pay-per-Feature", "Free Only"], N, p=[0.32, 0.22, 0.20, 0.16, 0.10])

# Monthly budget based on income + health concern
budget_base = np.where(income_band == "Below ₹15k", 100,
              np.where(income_band == "₹15k-₹30k", 199,
              np.where(income_band == "₹30k-₹60k", 299,
              np.where(income_band == "₹60k-₹1L", 499, 799))))
budget_noise = np.random.normal(0, 60, N).astype(int)
monthly_budget = np.clip(budget_base + budget_noise, 49, 1999)

willingness_to_pay = np.where(monthly_budget < 200, "Below ₹200",
                    np.where(monthly_budget < 350, "₹200-₹349",
                    np.where(monthly_budget < 600, "₹350-₹599", "₹600+")))

subscription_interest = np.random.choice(["Yes", "No"], N, p=[0.58, 0.42])
family_plan_interest = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
discount_sensitivity = np.random.choice(["Low", "Medium", "High"], N, p=[0.22, 0.40, 0.38])

# Sign-up intent: derived from key indicators
sign_up_score = (
    (health_concern_level == "High").astype(int) * 2 +
    (trust_app_reco == "High").astype(int) * 2 +
    (trust_ai_diet == "High").astype(int) +
    (interest_personalised == "Yes").astype(int) +
    (fitness_goal == "Weight Loss").astype(int) +
    (doctor_recommendation == "Yes").astype(int) +
    (data_sharing_comfort == "Comfortable").astype(int) +
    np.random.randint(0, 3, N)
)
sign_up_intent = np.where(sign_up_score >= 7, "High", np.where(sign_up_score >= 4, "Medium", "Low"))
likely_to_subscribe = np.where(sign_up_score >= 6, "Yes", "No")
recommend_likelihood = np.clip(
    (sign_up_score * 1.0 + np.random.normal(0, 1.5, N)).round(1), 1, 10
)

# --- Engineered Features ---
bmi_num = bmi.copy()
health_risk_score = (
    (bmi_num > 27).astype(int) * 2 +
    (health_conditions != "None").astype(int) * 2 +
    (family_health_history != "None").astype(int) +
    (junk_food_freq == "Daily").astype(int) +
    (sugar_intake == "High").astype(int) +
    (activity_level == "Sedentary").astype(int)
).clip(0, 10)

diet_quality_score = (
    (eating_habit_quality == "Good").astype(int) * 2 +
    (eating_habit_quality == "Excellent").astype(int) * 3 +
    (meals_per_day == 3).astype(int) +
    (water_intake == ">3L").astype(int) * 2 +
    (junk_food_freq == "Rarely").astype(int) * 2 +
    (meal_timing_regularity == "Regular").astype(int) * 2
).clip(0, 10)

fitness_readiness_score = (
    (activity_level == "Moderately Active").astype(int) * 2 +
    (activity_level == "Highly Active").astype(int) * 3 +
    (exercise_consistency == "5+x/week").astype(int) * 2 +
    (exercise_consistency == "2-3x/week").astype(int) +
    (sleep_hours == "7-8 hrs").astype(int) * 2 +
    (health_motivation == "High").astype(int) * 2
).clip(0, 10)

digital_trust_score = (
    (trust_app_reco == "High").astype(int) * 3 +
    (trust_ai_diet == "High").astype(int) * 3 +
    (data_sharing_comfort == "Comfortable").astype(int) * 2 +
    (past_health_app == "Yes").astype(int) * 2
).clip(0, 10)

app_engagement_score = (
    (reminder_preference != "None").astype(int) +
    (interest_personalised == "Yes").astype(int) * 2 +
    (interest_recipe_lib == "Yes").astype(int) +
    (interest_expert_chat == "Yes").astype(int) +
    (interest_ai_form == "Yes").astype(int) +
    (interest_ad_free == "Yes").astype(int) +
    (health_motivation == "High").astype(int) * 2
).clip(0, 10)

price_sensitivity_score = (
    (discount_sensitivity == "High").astype(int) * 3 +
    (app_stop_reason == "Too Expensive").astype(int) * 3 +
    (barrier_to_app == "Too Expensive").astype(int) * 2 +
    (pricing_model == "Free Only").astype(int) * 3
).clip(0, 10)

subscription_potential = (
    (sign_up_intent == "High").astype(int) * 3 +
    (subscription_interest == "Yes").astype(int) * 2 +
    (paid_health_before == "Yes").astype(int) * 2 +
    (digital_trust_score > 6).astype(int) * 2 +
    (health_concern_level == "High").astype(int)
).clip(0, 10)

personalisation_need = (
    (health_conditions != "None").astype(int) * 2 +
    (food_allergies != "None").astype(int) * 2 +
    (fitness_goal != "General Wellness").astype(int) +
    (bmi_num > 27).astype(int) * 2 +
    (interest_personalised == "Yes").astype(int) * 2 +
    (diet_quality_score < 5).astype(int)
).clip(0, 10)

motivation_risk_score = (
    (motivation_dropoff == "Lack of Time").astype(int) * 2 +
    (diet_break_response == "Give up").astype(int) * 3 +
    (exercise_consistency == "Never").astype(int) * 2 +
    (health_motivation == "Low").astype(int) * 3
).clip(0, 10)

customer_value_score = (
    (monthly_budget > 400).astype(int) * 2 +
    (paid_health_before == "Yes").astype(int) * 2 +
    (subscription_potential > 6).astype(int) * 2 +
    (recommend_likelihood > 7).astype(int) * 2 +
    (family_plan_interest == "Yes").astype(int)
).clip(0, 10)

retention_risk_score = (
    (motivation_risk_score > 6).astype(int) * 2 +
    (app_stop_reason != "No Reason").astype(int) * 2 +
    (digital_trust_score < 4).astype(int) * 2 +
    (discount_sensitivity == "High").astype(int) +
    (barrier_to_app == "Too Expensive").astype(int) * 2
).clip(0, 10)

# Persona
def assign_persona(i):
    if age_groups[i].startswith("18-24") and monthly_budget[i] < 250:
        return "Budget-Conscious Student"
    elif fitness_goal[i] == "Weight Loss" and bmi_num[i] > 27:
        return "Weight-Loss Seeker"
    elif age_groups[i].startswith("25-34") and activity_level[i] in ["Moderately Active", "Highly Active"]:
        return "Fitness-Focused Young Professional"
    elif occupation[i] == "Homemaker":
        return "Diet-Conscious Homemaker"
    elif health_conditions[i] != "None":
        return "Medical Condition Manager"
    elif customer_value_score[i] > 7:
        return "Premium Wellness User"
    elif digital_trust_score[i] < 4:
        return "Low-Trust App User"
    elif lifestyle_risk[i] == "High":
        return "High-Risk Lifestyle User"
    elif family_plan_interest[i] == "Yes" and household_size[i] >= 4:
        return "Family Plan Prospect"
    else:
        return "General Wellness Seeker"

user_persona = np.array([assign_persona(i) for i in range(N)])

def recommend_plan(i):
    if monthly_budget[i] < 200:
        return "Free Plan"
    elif monthly_budget[i] < 350:
        return "Basic ₹199/month"
    elif monthly_budget[i] < 600:
        return "Standard ₹299/month"
    elif monthly_budget[i] >= 600:
        return "Premium ₹499/month"
    return "Free Plan"

plan_recommendation = np.array([recommend_plan(i) for i in range(N)])

# --- Build DataFrame ---
df = pd.DataFrame({
    "Respondent_ID": [f"NQ{str(i+1).zfill(4)}" for i in range(N)],
    "Age_Group": age_groups,
    "Gender": gender,
    "City_Tier": city_tier,
    "Occupation": occupation,
    "Income_Band": income_band,
    "Education": education,
    "Household_Size": household_size,
    "Height_cm": height,
    "Weight_kg": weight,
    "BMI": bmi,
    "BMI_Category": bmi_category,
    "Health_Conditions": health_conditions,
    "Food_Allergies": food_allergies,
    "Medical_Restrictions": medical_restrictions,
    "Doctor_Recommendation": doctor_recommendation,
    "Family_Health_History": family_health_history,
    "Health_Concern_Level": health_concern_level,
    "Food_Preference": food_preference,
    "Meals_Per_Day": meals_per_day,
    "Eating_Outside_Freq": eating_outside_freq,
    "Eating_Habit_Quality": eating_habit_quality,
    "Preferred_Cuisine": preferred_cuisine,
    "Meal_Timing_Regularity": meal_timing_regularity,
    "Junk_Food_Freq": junk_food_freq,
    "Sugar_Intake": sugar_intake,
    "Water_Intake": water_intake,
    "Diet_Break_Response": diet_break_response,
    "Activity_Level": activity_level,
    "Fitness_Goal": fitness_goal,
    "Workout_Preference": workout_preference,
    "Sleep_Hours": sleep_hours,
    "Health_Motivation": health_motivation,
    "Exercise_Consistency": exercise_consistency,
    "Workout_Location": workout_location,
    "Motivation_Dropoff": motivation_dropoff,
    "Lifestyle_Risk": lifestyle_risk,
    "Health_Personality": health_personality,
    "Data_Sharing_Comfort": data_sharing_comfort,
    "Health_Influencer": health_influencer,
    "App_Stop_Reason": app_stop_reason,
    "Trust_App_Reco": trust_app_reco,
    "Trust_AI_Diet": trust_ai_diet,
    "Reminder_Preference": reminder_preference,
    "Barrier_To_App": barrier_to_app,
    "Notif_Channel": notif_channel,
    "Past_Health_App": past_health_app,
    "Services_Wanted": services_wanted,
    "Meal_Plan_Type": meal_plan_type,
    "Paid_Health_Before": paid_health_before,
    "Upgrade_Trigger": upgrade_trigger,
    "Interest_Expert_Chat": interest_expert_chat,
    "Interest_Recipe_Lib": interest_recipe_lib,
    "Interest_Travel_Diet": interest_travel_diet,
    "Interest_AI_Form": interest_ai_form,
    "Interest_Ad_Free": interest_ad_free,
    "Interest_Personalised": interest_personalised,
    "Past_Health_Spend": past_health_spend,
    "Pricing_Model": pricing_model,
    "Monthly_Budget": monthly_budget,
    "Willingness_To_Pay": willingness_to_pay,
    "Subscription_Interest": subscription_interest,
    "Family_Plan_Interest": family_plan_interest,
    "Discount_Sensitivity": discount_sensitivity,
    "Sign_Up_Intent": sign_up_intent,
    "Likely_To_Subscribe": likely_to_subscribe,
    "Recommend_Likelihood": np.round(recommend_likelihood, 1),
    # Engineered
    "Health_Risk_Score": health_risk_score,
    "Diet_Quality_Score": diet_quality_score,
    "Fitness_Readiness_Score": fitness_readiness_score,
    "Digital_Trust_Score": digital_trust_score,
    "App_Engagement_Score": app_engagement_score,
    "Price_Sensitivity_Score": price_sensitivity_score,
    "Subscription_Potential": subscription_potential,
    "Personalisation_Need": personalisation_need,
    "Motivation_Risk_Score": motivation_risk_score,
    "Customer_Value_Score": customer_value_score,
    "Retention_Risk_Score": retention_risk_score,
    "User_Persona": user_persona,
    "Plan_Recommendation": plan_recommendation,
})

df.to_csv("/home/claude/nourishiq/nourishiq_dataset.csv", index=False)
print("Dataset saved:", df.shape)
print(df.dtypes.value_counts())
print(df["Sign_Up_Intent"].value_counts())
print(df["Likely_To_Subscribe"].value_counts())
