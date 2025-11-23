f = [-2;-3;-1];% 目标函数系数 (注意：linprog默认求最小值，所以最大化要加负号)
A = [-1,-4,-2;-3,-2,1];% 不等式约束系数矩阵
b = [-8; -6];% 不等式约束右侧值
Aeq = [];% 等式约束系数矩阵
beq = [];% 等式约束右侧值
lb = [0;0;0];% 变量下界
ub = [];% 变量上界
[x, fval, exitflag] = linprog(-f, A, b, Aeq, beq, lb, ub);% 求解线性规划

if exitflag > 0
    fprintf('最优解找到！\n');
    fprintf('x1 = %.4f\n', x(1));
    fprintf('x2 = %.4f\n', x(2));
    fprintf('x3 = %.4f\n', x(3));
    fprintf('最优目标函数值 = %.4f\n', -fval);% 注意取负号
else
    fprintf('未找到最优解，退出标志: %d\n', exitflag);
end