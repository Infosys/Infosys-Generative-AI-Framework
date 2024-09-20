def calculate_fibonacci(n):
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


number = 10
fibonacci_sequence = calculate_fibonacci(number)
print(
    f"The first {number} numbers in the Fibonacci sequence are: {fibonacci_sequence}")
