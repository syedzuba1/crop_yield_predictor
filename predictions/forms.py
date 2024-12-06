from django import forms

class EconomicImpactForm(forms.Form):
    age = forms.FloatField(label="Age (From 1 to 60)")
    gender = forms.CharField(label="Gender (Male or Female)")


class AdaptationPredictionForm(forms.Form):
    total_education_facilities = forms.FloatField(label="Total Education Facilities")
    escapee_rate = forms.FloatField(label="Escapee Rate (From 0 to 1)")
    mental_illness_rate = forms.FloatField(label="Mental Illness Rate (From 0 to 1)")

class CrimePredictionForm(forms.Form):
    state = forms.CharField(label="State")
    crime_type = forms.CharField(label="Crime Type (Choose from the graph)")

class BudgetPredictionForm(forms.Form):
    state_ut = forms.CharField(label='State/UT')
    year = forms.IntegerField(label='Base Year')
    num_years = forms.IntegerField(label='Number of Years to Predict')