
x1 = [0.1,0.2,0.4,0.9,1.2]; % 人均专著 >
x2 = [5,6,7,10,2]; % 生师比 [5,6]
x3 = [5000,6000,7000,10000,400]; % 科研经费 >
x4 = [4.7,5.6,6.7,2.3,1.8]; % 逾期毕业率 <

% step2
y2 = zeros(size(x2));
y2(x2<=2) = 0;
y2(x2>2 & x2<=5) = ((x2(x2>2 & x2<=5)-2)*1/3);
y2(x2>5 & x2<=6) = 1;
y2(x2>6 & x2<=12) = (1-(x2(x2>6 & x2<=12)-6)*1/6);
y2(x2>12) = 0;
y2 % 归一化处理x2

y4 = zeros(size(x4));
y4 = 1./x4;
y4 % 倒数处理x4


% step3
x1_1 = norm(x1);
x1_norm = x1./norm(x1)

x2_1 = norm(y2);
x2_norm = y2./norm(y2)

x3_1 = norm(x3);
x3_norm = x3./norm(x3)

x4_1 = norm(y4);
x4_norm = y4./norm(y4)

% step4
school_best = [max(x1_norm),max(x2_norm),max(x3_norm),max(x4_norm)];
school_worst = [min(x1_norm),min(x2_norm),min(x3_norm),min(x4_norm)];

% step5
X = [x1_norm(1),x2_norm(1),x3_norm(1),x4_norm(1)
    x1_norm(2),x2_norm(2),x3_norm(2),x4_norm(2)
    x1_norm(3),x2_norm(3),x3_norm(3),x4_norm(3)
    x1_norm(4),x2_norm(4),x3_norm(4),x4_norm(4)
    x1_norm(5),x2_norm(5),x3_norm(5),x4_norm(5)]

w = [0.2,0.3,0.4,0.1]; % 权重

n = size(X,1);
distance_to_best = [n,1];
distance_to_worst = [n,1];

for i = 1:n
    distance_to_best(i) = sqrt(sum(w(1,:).*(school_best(1,:) - X(i,:)).^2));
    distance_to_worst(i) = sqrt(sum(w(1,:).*(school_worst(1,:) - X(i,:)).^2));
end
% 分别求每个学校到最优方案和最劣方案的距离

X_final = distance_to_worst./(distance_to_best + distance_to_worst)
% 求最终得分