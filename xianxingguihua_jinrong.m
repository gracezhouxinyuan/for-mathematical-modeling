% 初始化参数与空列表
a = 0;
profit_list = [];
a_list = [];

% 循环求解线性规划，a从0递增到0.05（步长0.001）
while a < 0.05
    % 目标函数系数（Python中为-c，MATLAB linprog默认求最小值，直接对应）
    c = [-0.05, -0.27, -0.19, -0.185, -0.185];
    
    % 不等式约束矩阵A（4行5列，第一列全0，后四列对角阵）
    A = [zeros(4,1), diag([0.025, 0.015, 0.055, 0.026])];
    
    % 不等式约束常数项b（4行1列，全为a）
    b = a * ones(4,1);
    
    % 等式约束矩阵与常数项
    Aeq = [1, 1.01, 1.02, 1.045, 1.065];
    beq = 1;
    
    % 变量边界（所有变量x≥0，无上界）
    lb = zeros(5,1);  % 下界：全0
    ub = Inf(5,1);    % 上界：无穷大
    bounds = [lb, ub];
    
    % 调用linprog求解线性规划（options消除默认输出）
    options = optimoptions('linprog', 'Display', 'off');
    [~, fval] = linprog(c, A, b, Aeq, beq, lb, ub, options);
    
    % 计算利润（目标函数最小值的相反数）并存储数据
    profit = -fval;
    profit_list = [profit_list, profit];
    a_list = [a_list, a];
    
    % 更新a的值
    a = a + 0.001;
end

% 绘制利润随a变化的曲线
figure('Position', [100, 100, 800, 560]);  % 对应figsize=(10,7)
plot(a_list, profit_list);
xlabel('a');
ylabel('profit');
grid on;  % 添加网格线，便于读数
title('Profit vs Parameter a');  % 增加标题，提升可读性