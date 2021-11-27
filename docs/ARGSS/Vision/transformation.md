![img](media/transformation/v2-fc062f8ea80bec759403e46e601c58ed_1440w.jpg)

总结来说就是相似三角形，$\frac{x_{c}}{x}=\frac{y_{c}}{y}=\frac{z_{c}}{f}$，通过变换可得$x=\frac{x_{c}}{z_{c}} f, y=\frac{y_{c}}{z_{c}} f$，由此得其增广形式
$$
\left(\begin{array}{l}
x \\
y \\
1
\end{array}\right)=\frac{1}{z_{c}}\left(\begin{array}{llll}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{array}\right)\left(\begin{array}{c}
x_{c} \\
y_{c} \\
z_{c} \\
1
\end{array}\right)
$$
相机坐标系到机械臂坐标就是一个平移加旋转，一个矩阵搞定的，一个不够就连乘

如下是绕$z$轴旋转
$$
\left(\begin{array}{l}
x_{c} \\
y_{c} \\
z_{c} \\
1
\end{array}\right)=\left(\begin{array}{ccc}
\cos \alpha & -\sin \alpha & 0 & 0\\
\sin \alpha & \cos \alpha & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{array}\right)\left(\begin{array}{l}
x_{w} \\
y_{w} \\
z_{w} \\
1
\end{array}\right)
$$