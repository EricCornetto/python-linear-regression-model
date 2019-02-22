# ===========Linear Regression Model=========================================
#  *   Linear Regression Algorithm.
# =============================================================================
#  *   Created By Eric Theodore Cornetto(Ida Bagus Dwi Putra Purnawa).
#  *   Github (https://github.com/EricCornetto).
# =============================================================================
#  *   GNU General Public License v3.0.
# =============================================================================
#             Python Machine Learning
# =============================================================================
# Linear Algorithm
# Y = a + bx
class LinearRegression():
    #fitting data
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n_train = len(self.x_train)
    #find sigma X
    def sigmaX(self):
        result = np.sum(self.x_train)
        return result

    #find sigma Y
    def sigmaY(self):
        result = np.sum(self.y_train)
        return result

    #find sigma XY
    def sigmaXY(self):
        multiply = np.multiply(self.x_train,self.y_train)
        result = np.sum(multiply)
        return result

    #find sigma X^2
    def sigmaSQRX(self):
        sqr = np.square(self.x_train)
        result = np.sum(sqr)
        return result

    #find sigma Y^2
    def sigmaSQRY(self):
        sqr = np.square(self.y_train)
        result = np.sum(sqr)
        return result

    #find A
    def findA(self):
        sigmaX = self.sigmaX()
        sigmaY = self.sigmaY()
        sigmaXY = self.sigmaXY()
        sigmaSQRX = self.sigmaSQRX()
        sigmaSQRY = self.sigmaSQRY()
        sigmaXSQR = np.square(sigmaX)
        result = ((sigmaY)*(sigmaSQRX) - (sigmaX)*(sigmaXY)) / ((self.n_train)*(sigmaSQRX) - (sigmaXSQR))
        return result

    #find B
    def findB(self):
        sigmaX = self.sigmaX()
        sigmaY = self.sigmaY()
        sigmaXY = self.sigmaXY()
        sigmaSQRX = self.sigmaSQRX()
        sigmaSQRY = self.sigmaSQRY()
        sigmaXSQR = np.square(sigmaX)
        result = ((self.n_train)*(sigmaXY) - (sigmaX)*(sigmaY)) / ((self.n_train)*(sigmaSQRX) - (sigmaXSQR))
        return result

    #Prediction
    def predict(self,x_data):
        valueA = self.findA()
        valueB = self.findB()
        y_pred = valueA + (valueB*x_data)
        return y_pred
