## sum (0,1)
Definition: arr = [$a_1, a_2, \cdots, a_n$], $0<a_i<1$, $p$ is the probability of $\sum_{i=1}^n{a_i} < 1$
$$
\begin{aligned}
1-p &= \int_{a_1=0}^{a_1=1} \int_{a_2=0}^{a_2=1} \cdots \int_{a_{n-1}=0}^{a_{n-1}=1-\sum_{i=1}^{n-2}a_i} {1-\sum_{i=1}^{n-1} {a_i}} {\mathrm{d} a_{n-1} \cdots \mathrm{d}a_2 \mathrm{d}a_1 }\\
&=\int_{a_1=0}^{a_1=1} \int_{a_2=0}^{a_2=1} \cdots \int_{a_{n-1}=0}^{a_{n-1}=1-\sum_{i=1}^{n-2}a_i} {1-\sum_{i=1}^{n-2} {a_i} - a_{n-1}} {\mathrm{d} a_{n-1} \cdots \mathrm{d}a_2 \mathrm{d}a_1}\\
&=\int_{a_1=0}^{a_1=1} 
    \int_{a_2=0}^{a_2=1} \cdots 
        \int_{a_{n-2}=1}^{a_{n-2}=1-\sum_{i=1}^{n-3}a_i}
        \left.(1-\sum_{i=1}^{n-2}a_i)a_{n-1} - \frac{a_{n-1}^2}{2} \right|_{a_{n-1}=0}^{a_{n-1}=1-\sum_{i=1}^{n-2}a_i} 
        \mathrm{d}a_{n-2} \cdots 
    \mathrm{d}a_2
\mathrm{d}a_1 \\

&=\int_{a_1=0}^{a_1=1} 
    \int_{a_2=0}^{a_2=1} \cdots 
        \int_{a_{n-2}=1}^{a_{n-2}=1-\sum_{i=1}^{n-3}a_i}
        \frac{(1-\sum_{i=1}^{n-2}a_i)^2}{2}  
        \mathrm{d}a_{n-2} \cdots 
    \mathrm{d}a_2 
\mathrm{d}a_1 \\

&=\int_{a_1=0}^{a_1=1} 
    \int_{a_2=0}^{a_2=1} \cdots 
        \int_{a_{n-3}=1}^{a_{n-3}=1-\sum_{i=1}^{n-4}a_i}
        \left.-\frac{(1-\sum_{i=1}^{n-3}a_i-a_{n-2})^3}{2*3}  \right|_{a_{n-2}=0}^{a_{n-2}=1-\sum_{i=1}^{n-3}a_i} 
        \mathrm{d}a_{n-3} \cdots 
    \mathrm{d}a_2
\mathrm{d}a_1 \\
&=\int_{a_1=0}^{a_1=1} 
    \int_{a_2=0}^{a_2=1} \cdots 
        \int_{a_{n-3}=1}^{a_{n-3}=1-\sum_{i=1}^{n-4}a_i}
        \frac{(1-\sum_{i=1}^{n-3}a_i)^3}{2*3} 
        \mathrm{d}a_{n-3} \cdots 
    \mathrm{d}a_2
\mathrm{d}a_1 \\
&=\cdots\\
&=\left.-\frac{(1-a_1)^n}{n!}  \right|_0^1\\
&=\frac{1}{n!}\\
p&=1-\frac{1}{n!}
\end{aligned}
$$ 
