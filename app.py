import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
