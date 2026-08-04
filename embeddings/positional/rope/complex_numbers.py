import numpy as np


vector = np.complex128(3 + 4j) # создаем комплексное число
vector_real = vector.real # получаем действительную часть 3
vector_img = vector.imag # получаем мнимую часть 4
r = np.abs(vector) # вычисляем норму
theta = np.atan2(vector.imag, vector.real) # а тут вычисляем направление (theta)

complex_number = r * (np.exp(theta * 1j)) # с помощью формулы эйлера переходим в алгебраическую форму комплексного числа
# print(complex_number)

# -------
vector2 = np.complex128(2 + 1j) # создаем второе число
vector3 = vector * vector2 # переумножаем
r3 = np.abs(vector3) # вычисляем норму
theta3 = np.atan2(vector3.imag, vector3.real) # а тут вычисляем направление (theta)

# print(r3, theta3)

# -----
phi = np.radians(45)
print(f"initial r = {r}, theta = {theta}")

# повернем 3 + 4i на 45 градусов
new_vector = vector * np.exp(1j * phi)
print(f"new r = {np.abs(new_vector)}, theta = {np.atan2(new_vector.imag, new_vector.real)}, dif: {theta - np.atan2(new_vector.imag, new_vector.real)}")
# видим, что норма не поменялась, то есть вектор просто поменял свое направление

# -----
print("-" * 100)
numbers = np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype = np.complex128)
print(numbers)
# давайте повернем все числа на 67 градусов
phi = np.radians(67)
numbers_67 = np.abs(numbers) * np.exp(1j * (phi + np.atan2(numbers.imag, numbers.real)))
number_67_2 = numbers * np.exp(1j * phi)
print(numbers_67, "|", number_67_2)


