import math
from decimal import Decimal, getcontext

# Set the precision you desire
getcontext().prec = 2000  # for example, 50 decimal places

def calculate_pi_decimal(num_terms):
    # Use Decimal for high precision calculations
    pi_inverse = Decimal(0)
    for k in range(num_terms):
        numerator = ((-1) ** k) * Decimal(math.factorial(6 * k)) * (13591409 + 545140134 * k)
        denominator = Decimal(math.factorial(3 * k) * (math.factorial(k) ** 3)) * (640320 ** (3 * k + Decimal(1.5)))
        pi_inverse += numerator / denominator
    pi_approx = Decimal(1) / (pi_inverse * 12)
    return pi_approx

# Number of terms for the approximation
num_terms = 100  # You can increase this number for more precision

# Calculate pi with higher precision
pi_approx_decimal = calculate_pi_decimal(num_terms)

print(f"Approximated value of pi using {num_terms} terms: {pi_approx_decimal}")