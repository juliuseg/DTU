# Code written in collaboration with Chat-gpt4

from scipy import stats
import numpy as np


crossBaseline = np.array([285.11665996463296, 295.28872737576336, 299.26614426337824, 294.7764593187116, 289.31694283914237, 276.44039174593485, 305.6796230907934, 287.6857190629241, 288.76788750203525, 290.7162355656017])
crossRegression = np.array([19.012424092830955, 21.534770759785907, 23.088864691705542, 19.09145673050284, 24.086296046321614, 19.76065574443629, 18.848628756393843, 22.19284243318082, 20.691475044736126, 19.763062448967503])
crossANN = np.array([18.825311335721608, 21.692763739635907, 23.041483276685653, 19.129276466472614, 23.936057925382354, 19.920603026170618, 18.55740769465106, 22.23483631755047, 20.672880519201566, 19.70940152882168])

classificationBaselineErrors = np.array([0.5193312434691746, 0.5182863113897597, 0.5015673981191222, 0.5005224660397074, 0.5203761755485894, 0.5067920585161965, 0.502612330198537, 0.5182863113897597, 0.5010460251046025, 0.5083682008368201])
classificationRegressionErrors = np.array([0.054336468129571575, 0.04597701149425287, 0.04179728317659352, 0.0522466039707419, 0.055381400208986416, 0.054336468129571575, 0.04597701149425287, 0.04493207941483804, 0.038702928870292884, 0.05439330543933055])
classificationANNErrors = np.array([0.053291536050156685, 0.05120167189132707, 0.043887147335423204, 0.053291536050156685, 0.056426332288401215, 0.0543364681295716, 0.04806687565308254, 0.04597701149425293, 0.03870292887029292, 0.05543933054393302])

A = "Classification baseline"
B = "Classification regression"
C = "Classification ANN"

a = classificationBaselineErrors 
b = classificationRegressionErrors
c = classificationANNErrors

# Pairwise T-tests
t_stat_ab, p_value_ab = stats.ttest_rel(a, b)
t_stat_ac, p_value_ac = stats.ttest_rel(a, c)
t_stat_bc, p_value_bc = stats.ttest_rel(b, c)

# Confidence Intervals for the differences in means
alpha = 0.05
df = len(a) - 1
confidence_level = 1 - alpha

# Calculate standard error of the mean difference for scale
sem_ab = stats.sem(a - b)
sem_ac = stats.sem(a - c)
sem_bc = stats.sem(b - c)

ci_low_ab, ci_high_ab = stats.t.interval(confidence_level, df, loc=np.mean(a - b), scale=sem_ab)
ci_low_ac, ci_high_ac = stats.t.interval(confidence_level, df, loc=np.mean(a - c), scale=sem_ac)
ci_low_bc, ci_high_bc = stats.t.interval(confidence_level, df, loc=np.mean(b - c), scale=sem_bc)

# Print the results
print(f"{A} model vs {B} model: p-value = {p_value_ab},\n CI = ({ci_low_ab}, {ci_high_ab})")
print(f"{A} model vs {C} model: p-value = {p_value_ac},\n CI = ({ci_low_ac}, {ci_high_ac})")
print(f"{B} model vs {C} model: p-value = {p_value_bc},\n CI = ({ci_low_bc}, {ci_high_bc})")

print("\n")

# Null hypothesis interpretation
if p_value_ab < alpha:
    print(f"Reject H0 for {A} model vs {B} model: The difference in performance is statistically significant.")
else:
    print(f"Fail to reject H0 for {A} model vs {B} model: No significant difference in performance.")

print("")

if p_value_ac < alpha:
    print(f"Reject H0 for {A} model vs {C} model: The difference in performance is statistically significant.")
else:
    print(f"Fail to reject H0 for {A} model vs {C} model: No significant difference in performance.")

print("")


if p_value_bc < alpha:
    print(f"Reject H0 for {B} model vs {C} model: The difference in performance is statistically significant.")
else:
    print(f"Fail to reject H0 for {B} model vs {C} model: No significant difference in performance.")
