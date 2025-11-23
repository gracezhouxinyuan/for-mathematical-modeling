% 数据点
x = [0, 1, 2.5];
y = [0, 1, 2];

% 最小二乘法计算
n = length(x);
sum_x = sum(x);
sum_y = sum(y);
sum_xy = sum(x .* y);
sum_x2 = sum(x .^ 2);

% 计算斜率和截距
k = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2);
b = (sum_x2 * sum_y - sum_x * sum_xy) / (n * sum_x2 - sum_x^2);

% 输出结果
fprintf('最小二乘法线性回归结果：\n');
fprintf('斜率 k = %.4f\n', k);
fprintf('截距 b = %.4f\n', b);
fprintf('回归方程: y = %.4fx + %.4f\n', k, b);