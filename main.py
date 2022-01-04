import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import plotly.express as px 
st.title("Machine Learning classifiers")

st.write("""
# Explore the different classifier 
""")

dataset_name = st.sidebar.selectbox("Select the dataset",("Iris Dataset","Breast Cancer","Wine dataset"))
#st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select the classifier",("k-nearest neighbors",
                                                                "Support Vector Classification",
                                                                "Gradient Boosting",
                                                                "AdaBoost classifier",
                                                                "Multi-layer Perceptron classifier",
                                                                "Random Forest",
                                                                ))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y
    

X, y = get_dataset(dataset_name)


st.write("Selected dataset Name:",dataset_name)
st.write("Shape of the dataset:",X.shape)
st.write("Number of Output classes:",len(np.unique(y)))

# adding parameter for different classifier
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "k-nearest neighbors":
        K_kn = st.sidebar.slider("Number of neighbors: ",1,50,value=5,step=5)
        weights_kn = st.sidebar.selectbox("Select the weight:",("uniform","distance"),index=0)
        algorithm_kn = st.sidebar.selectbox("Select the algorithm:",("auto","ball_tree","kd_tree","brute"),index=0)
        leaf_size_kn = st.sidebar.slider("Select the leaf size: ",10,100,value=30,step=10)
        p_kn = st.sidebar.slider("Select the value of P: ",1,2,value=2)
        params["leaf_size_kn"] = leaf_size_kn
        params["weights_kn"] = weights_kn
        params["K_kn"] = K_kn
        params["algorithm_kn"] = algorithm_kn
        params["p_kn"] = p_kn
    elif clf_name == "Support Vector Classification":
        C_svc= st.sidebar.slider("C",0.01,10.0,value=1.0)
        kernel_svc = st.sidebar.selectbox("Select the kernal",("linear","poly","rbf","sigmoid"),index=2)
        if kernel_svc == "poly":
            degree_svc = st.sidebar.slider("Select the Degree ",0,15,value=3)
            params["degree_svc"] = degree_svc
        elif kernel_svc == "poly" or "rbf" or "sigmoid":
            gamma_svc = st.sidebar.selectbox("Select the gamma ",("scale","auto"),index=0)
            params["gamma_svc"] = gamma_svc
        cache_size_svc = st.sidebar.slider("Select the cache Size:",100,500,value=200)
        params["cache_size_svc"] = cache_size_svc
        params["C_svc"] = C_svc
        params["kernel_svc"] = kernel_svc
    elif clf_name == "Gradient Boosting":
        loss = st.sidebar.selectbox("Select the loss function: ",("deviance","exponential"),index=0)
        learning_rate_grad = st.sidebar.slider("Learning rate: ",0.01,1.0,value=0.1)
        n_estimators_grad = st.sidebar.slider("n_estimators: ", 100, 300,value=100)
        subsample_grad = st.sidebar.slider("Select subsample: ", 0.1,1.0,value=1.0)
        criterion_grad = st.sidebar.selectbox("Select the criterion: ", ("friedman_mse","squared_error","mse","mae"),index=0)
        max_depth_grad = st.sidebar.slider("Select max_depth_grad: ",1,5,value=3,step=1)
        params["learning_rate_grad"] = learning_rate_grad
        params["n_estimator_grad"] = n_estimators_grad
        params["subsample_grad"] = subsample_grad
        params["criterion_grad"] = criterion_grad
        params["max_depth_grad"] = max_depth_grad
        params["loss_grad"] = loss
    elif clf_name == "AdaBoost classifier":
        n_estimators_ada = st.sidebar.slider("Select number of estimator: ",10,100,value=50,step=10)
        learning_rate_ada = st.sidebar.slider("Select the learning rate: ",0.01,1.0,value=1.0)
        algorithm_ada = st.sidebar.selectbox("Select the algorithm: ",("SAMME","SAMME.R"),index=1)
        params["n_estimators_ada"] = n_estimators_ada
        params["learning_rate_ada"] = learning_rate_ada
        params["algorithm_ada"] = algorithm_ada
    elif clf_name == "Multi-layer Perceptron classifier":
        activation_mlp = st.sidebar.selectbox("Select activation Function: ",("identity","logistic","tanh","relu"),index=3) 
        solver_mlp = st.sidebar.selectbox("Select the solver: ",("lbfgs","sgd","adam"),index=2)
        alpha_mlp = st.sidebar.slider("Select the alpha: ",0.0001,1.0,value=0.0001)
        if solver_mlp == "sgd":
            learning_rate_mlp = st.sidebar.selectbox("Select the learning rate :",("constant","invscaling","adaptive"),index=0)
            params["learning_rate_mlp"] = learning_rate_mlp
            learning_rate_init_mlp = st.sidebar.slider("Select the learning rate in integer:",0.0001,1.0,value=0.001)
            params["learning_rate_init_mlp"] = learning_rate_init_mlp 
            shuffle_mlp = st.sidebar.selectbox("Shuffle in each iteration:",(True,False),index=0)
            params["shuffle_mlp"] = shuffle_mlp
        if solver_mlp == "adam":
            learning_rate_init_mlp = st.sidebar.slider("Select the learning rate in integer:",0.0001,1.0,value=0.001)
            params["learning_rate_init_mlp"] = learning_rate_init_mlp
            shuffle_mlp = st.sidebar.selectbox("Shuffle in each iteration:",(True,False),index=0)
            params["shuffle_mlp"] = shuffle_mlp 
        params["activation_mlp"] = activation_mlp
        params["solver_mlp"] = solver_mlp
        params["alpha_mlp"] = alpha_mlp
    else:
        max_depth = st.sidebar.slider("Select maximum depth of the tree: ",1,15)
        n_estimator = st.sidebar.slider("Number of tree in the forest: ", 1,200,value=100)
        criterion = st.sidebar.selectbox("Select the criterion: ",("gini","entropy"),index=0)
        min_samples_split = st.sidebar.slider("Select Minimum number of sample required to split: ",0.1,1.0,value=1.0)
        max_features = st.sidebar.selectbox("Select the number of feature: ",("auto","sqrt","log2"),index=0)
        params["max_features"] = max_features
        params["criterion"] = criterion
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimator
        params["min_samples_split"] = min_samples_split
    return params

params = add_parameter_ui(classifier_name)

# getting the classifier
def get_classifier(clf_name, params):
    if clf_name == "k-nearest neighbors":
        clf = KNeighborsClassifier(n_neighbors=params["K_kn"],
                                   weights=params["weights_kn"],
                                   algorithm=params["algorithm_kn"],
                                   leaf_size=params["leaf_size_kn"],
                                   p=params["p_kn"])
    elif clf_name == "Support Vector Classification": 
        if params["kernel_svc"] == "linear":
            clf = SVC(C=params["C_svc"],kernel=params["kernel_svc"],
                                        cache_size=params["cache_size_svc"],
                                        random_state=12)
        elif params["kernel_svc"] == "poly":
            clf = SVC(C=params["C_svc"],kernel=params["kernel_svc"],
                                        degree=params["degree_svc"],
                                        cache_size=params["cache_size_svc"],
                                        random_state=12)
        elif params["kernel_svc"] == "poly" or "rbf" or "sigmoid":
            clf = SVC(C=params["C_svc"],kernel=params["kernel_svc"],
                                        gamma=params["gamma_svc"],
                                        cache_size=params["cache_size_svc"],
                                        random_state=12)
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(loss=params["loss_grad"],
                                            learning_rate=params["learning_rate_grad"],
                                            n_estimators=params["n_estimator_grad"],
                                            subsample=params["subsample_grad"],
                                            criterion=params["criterion_grad"],
                                            max_depth=params["max_depth_grad"])
    elif clf_name == "AdaBoost classifier":
        clf = AdaBoostClassifier(n_estimators=params["n_estimators_ada"],
                                learning_rate=params["learning_rate_ada"],
                                algorithm=params["algorithm_ada"],
                                random_state=12)
    elif clf_name == "Multi-layer Perceptron classifier":
        if params["solver_mlp"] == "sgd":
            clf = MLPClassifier(activation=params["activation_mlp"],
                                    solver=params["solver_mlp"],
                                    alpha=params["alpha_mlp"],
                                    learning_rate=params["learning_rate_mlp"],
                                    learning_rate_init=params["learning_rate_init_mlp"],
                                    shuffle=params["shuffle_mlp"],random_state=12
                                    )
        elif params["solver_mlp"] == "adam":
            clf = MLPClassifier(activation=params["activation_mlp"],
                                    solver=params["solver_mlp"],
                                    alpha=params["alpha_mlp"],
                                    learning_rate_init=params["learning_rate_init_mlp"],
                                    shuffle=params["shuffle_mlp"],random_state=12
                                    )
        else:
            clf = MLPClassifier(activation=params["activation_mlp"],
                                    solver=params["solver_mlp"],
                                    alpha=params["alpha_mlp"],
                                    random_state=12
                                    )
    else:
       clf = RandomForestClassifier(n_estimators=params["n_estimator"],
                                         max_depth=params["max_depth"],
                                         criterion=params["criterion"],
                                         min_samples_split=params["min_samples_split"],
                                         max_features=params["max_features"],
                                         random_state=12)
    return clf

clf = get_classifier(classifier_name, params)


# spliting the dataset in train and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting the model  
clf.fit(X_train,y_train)
# predicting the model on testing data
y_pred = clf.predict(X_test)
# checking the accuracy
acc = accuracy_score(y_test, y_pred)

st.write(f"Name of the classifier : {classifier_name}")
st.write(f"Accuracy of the classifier on testing data : {round(acc,5)*100} %")

# ploting the data
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0] # for zero dimension 
x2 = X_projected[:,1] # for first dimension



st.plotly_chart(px.scatter(data_frame=X_projected, x=x1, y=x2, color=y,),use_container_width=True)

# TODO
# add more parameter 
# add other classifier
# add feature scaling 