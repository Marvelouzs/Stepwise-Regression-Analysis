import pandas as pd
import numpy as np
import scipy.stats as stats
from numpy.linalg import inv
import streamlit as st

st.title('Stepwise Regression Factbook.csv')

# Functions
def clean_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(",", "").str.replace("(", "").str.replace(")", "").str.replace("$", "").astype(float)
    return df.dropna()

def create_X(df, *vars):
    n = df['GDP'].count()
    X = np.array([np.ones(n)] + [df[var] for var in vars]).T
    return X

def R_vars(df, *vars):
    n = df['GDP'].count()
    Y = np.array([df['GDP']]).T
    y_avg = np.average(Y)
    X = create_X(df, *vars)
    b = calculate_b(X, Y)
    y_hat = calculate_y_hat(X, b)
    Rb = calculate_Rb(y_hat, y_avg)
    return Rb, y_hat

def R_2vars(df, var1, var2):
    n = df['GDP'].count()
    Y = np.array([df['GDP']]).T
    y_avg = np.average(Y)

    X12 = np.array([np.ones(n), df[var1], df[var2]]).T
    b12 = np.matmul(np.matmul(inv(np.matmul(X12.T, X12)), X12.T), Y)
    y_hat12 = np.matmul(X12, b12) # Prediksi Y (GDP) menggunakan model 1
    Rb12 = np.matmul((y_hat12 - y_avg).T, (y_hat12 - y_avg))
    return Rb12, y_hat12

def calculate_b(X, Y): return np.matmul(np.matmul(inv(np.matmul(X.T, X)), X.T), Y)
def calculate_y_hat(X, b): return np.matmul(X, b)
def calculate_Rb(y_hat, y_avg): return np.matmul((y_hat - y_avg).T, (y_hat - y_avg))
def calculate_f(Rb, sse, n, k): return Rb / (sse / (n - k - 1))

# Read data
data = pd.read_csv("factbookv2.csv")

kolom1 = st.selectbox('X1', list(data.columns), index=list(data.columns).index('Exports'))
kolom2 = st.selectbox('X2', list(data.columns), index=list(data.columns).index('Imports'))
kolom3 = st.selectbox('X3', list(data.columns), index=list(data.columns).index('Industrial production growth rate'))
kolom4 = st.selectbox('X4', list(data.columns), index=list(data.columns).index('Investment'))
kolom5 = st.selectbox('X5', list(data.columns), index=list(data.columns).index('Unemployment rate'))

# Filter and process the data
df = clean_data(data[["GDP", kolom1, kolom2, kolom3, kolom4, kolom5]])

# Calculate Rb and y_hat for each model
Rb1, y_hat1 = R_vars(df, kolom1)
Rb2, y_hat2 = R_vars(df, kolom2)
Rb3, y_hat3 = R_vars(df, kolom3)
Rb4, y_hat4 = R_vars(df, kolom4)
Rb5, y_hat5 = R_vars(df, kolom5)

Y = np.array([df['GDP']]).T
n = df['GDP'].count()

# Cek b2 signifikan atau tidak
k = 1

sse2 = np.matmul((Y - y_hat2).T, (Y - y_hat2))
f2 = Rb2/(sse2 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.title('1 Variable Model')
st.write('Cek X2 signifikan atau tidak')
st.write('MSE Model 2:',sse2[0][0])
st.write('f dari Model 2:',f2[0][0])
st.write('f dari tabel:',f_table)

# Calculate Rb and y_hat for combined models
Rb21, y_hat21 = R_vars(df, kolom2, kolom1)
Rb23, y_hat23 = R_vars(df, kolom2, kolom3)
Rb24, y_hat24 = R_vars(df, kolom2, kolom4)
Rb25, y_hat25 = R_vars(df, kolom2, kolom5)

# Cek penambahan b1 signifikan atau tidak (R(b1 | b2))
# (R(b1 | b2) = R(b1, b2) - R(b2))
k = 2

sse21 = np.matmul((Y - y_hat21).T, (Y - y_hat21))
f21 = (Rb21 - Rb2)/(sse21 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.title('2 Combined Variable Models')
st.write('Cek penambahan X1 signifikan atau tidak (R(X1 | X2))')
st.write('MSE Model B1:',sse21[0][0])
st.write('f penambahan X1 pada Model B1:',f21[0][0])
st.write('f dari tabel:',f_table)

# Cek apakah b2 masih signifikan atau tidak, setelah penambahan b1
# (R(b2 | b1) = R(b2, b1) - R(b1))
sse21 = np.matmul((Y - y_hat21).T, (Y - y_hat21))
f21 = (Rb21 - Rb1)/(sse21 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.write('Cek apakah X2 masih signifikan atau tidak, setelah penambahan X1')
st.write('MSE Model B1:',sse21[0][0])
st.write('f X2 setelah ditambahkan X1 Model B1:',f21[0][0])
st.write('f dari tabel:',f_table)

# Calculate Rb and y_hat for 3-variable models
Rb213, y_hat213 = R_vars(df, kolom2, kolom1, kolom3)
Rb214, y_hat214 = R_vars(df, kolom2, kolom1, kolom4)
Rb215, y_hat215 = R_vars(df, kolom2, kolom1, kolom5)

# Cek penambahan b3 signifikan atau tidak (R(b3 | b2, b1))
# (R(b3 | b1, b2) = R(b1, b2, b3) - R(b1, b2))
k = 3

sse312 = np.matmul((Y - y_hat213).T, (Y - y_hat213))
f312 = (Rb213 - Rb21)/(sse312 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.title('3 Combined Variable Models')
st.write('Cek penambahan X3 signifikan atau tidak (R(X3 | X2, X1))')
st.write('MSE Model C1:',sse312[0][0])
st.write('f penambahan variabel 3 pada Model C1:',f312[0][0])
st.write('f dari tabel:',f_table)

# Cek apakah b1 masih signifikan setelah b3 ditambahkan
# (R(b1 | b2, b3) = R(b1, b2, b3) - R(b2, b3))
sse123 = np.matmul((Y - y_hat213).T, (Y - y_hat213))
f123 = (Rb213 - Rb23)/(sse123 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.write('Cek apakah X1 masih signifikan setelah X3 ditambahkan')
st.write('MSE Model C1:',sse123[0][0])
st.write('f penambahan variabel 3 pada Model C1:',f123[0][0])
st.write('f dari tabel:',f_table)

# Cek apakah b2 masih signifikan setelah b3 ditambahkan
# (R(b2 | b1, b3) = R(b1, b2, b3) - R(b1, b3))
Rb13, _ = R_2vars(df, kolom1, kolom3)
sse213 = np.matmul((Y - y_hat213).T, (Y - y_hat213))
f213 = (Rb213 - Rb13)/(sse213 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.write('Cek apakah X2 masih signifikan setelah X3 ditambahkan')
st.write('MSE Model C1:',sse213[0][0])
st.write('f penambahan variabel 3 pada Model C1:',f213[0][0])
st.write('f dari tabel:',f_table)

# Calculate Rb and y_hat for 4-variable models
Rb2134, y_hat2134 = R_vars(df, kolom2, kolom1, kolom3, kolom4)
Rb2135, y_hat2135 = R_vars(df, kolom2, kolom1, kolom3, kolom5)

# Cek penambahan b4 signifikan atau tidak (R(b4 | b1, b2, b3))
# (R(b4 | b1, b2, b3) = R(b1, b2, b3, b4) - R(b1, b2, b3))
k = 4

sse41234 = np.matmul((Y - y_hat2134).T, (Y - y_hat2134))
f41234 = (Rb2134 - Rb213)/(sse41234 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.title('4 Combined Variable Models')
st.write('Cek penambahan X4 signifikan atau tidak (R(X4 | X1, X2, X3))')
st.write('MSE Model D1:',sse41234[0][0])
st.write('f X4 setelah penambahan 4 variabel pada Model D1:',f41234[0][0])
st.write('f dari tabel:',f_table)

# Cek penambahan b5 signifikan atau tidak (R(X5 | X1, X2, X3))
# (R(X5 | X1, X2, X3) = R(X1, X2, X3, X5) - R(X1, X2, X3))
sse2135 = np.matmul((Y - y_hat2135).T, (Y - y_hat2135))
f2135 = (Rb2135 - Rb213)/(sse2135 / (n - k - 1))
f_table = stats.f.ppf(0.95, 1, n-k-1)
st.write('Cek penambahan X5 signifikan atau tidak (R(X5 | X1, X2, X3))')
st.write('MSE Model D2:',sse2135[0][0])
st.write('f X5 setelah penambahan 4 variabel pada Model D2:',f2135[0][0])
st.write('f dari tabel:',f_table)

# Calculate final model coefficients
bfinal = calculate_b(create_X(df, kolom1, kolom2, kolom3), np.array([df['GDP']]).T)
st.title('Hasil Akhir')
st.write('X0 =',bfinal[0][0])
st.write('X1 =',bfinal[1][0])
st.write('X2 =',bfinal[2][0])
st.write('X3 =',bfinal[3][0])
