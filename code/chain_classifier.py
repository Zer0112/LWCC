from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sp
from sklearn import clone
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearnex import patch_sklearn
import numpy as np

patch_sklearn(verbose=False)

class chain_configuration1(ClassifierChain):
    #legacy code

    def __init__(self, *, config_classifier=None, order=None, cv=None, random_state=None):
        self.config_classifier = config_classifier
        # self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, Y, **fit_params):
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        # check that the config_classifier list has the correct number of classifiers
        if len(self.config_classifier) != Y.shape[1]:
            raise ValueError(
                "The list of classifiers does not have the of labels")

        self.estimators_ = [clone(estimator)
                            for estimator in self.config_classifier]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(
                X_aug[:, : (X.shape[1] + chain_idx)], y, **fit_params)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx], y=y, cv=self.cv
                )
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result
        self.classes_ = [
            estimator.classes_ for chain_idx, estimator in enumerate(self.estimators_)
        ]
        return self



class chain_configuration_single_fit(ClassifierChain):
    """used to train a single classifier at a certain position and with a certain order.
    The trained classifier can be used as a pretrained classifier in the chain configuration"""
    
    def __init__(self, *, single_classifier, order=None,training_classifier_nr=0):
        """
        The function takes in a single classifier, an order, and a training classifier number.
        
        :param single_classifier: The classifier that will be used to train
        :param order: The order of the classifier in the chain
        :param training_classifier_nr: The position of the classifier that will be trained, defaults to 0
        (optional)
        """
        self.classifier = single_classifier
        self.order = order
        self.training_classifier_position = training_classifier_nr
    
    def fit(self, X, Y):
        """
        The function takes in the training data, and the training labels, and returns the trained
        classifier, the order of the classifiers, and the position of the classifier that is being trained.
        
        :param X: the input data
        :param Y: the target variable
        :return: The estimator, the order of the classifiers, and the position of the classifier being
        trained.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)
        
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = None
        
        
        Y_pred_chain = Y[:, self.order_]
        if sp.issparse(X):
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")
            X_aug = X_aug.tocsr()
        else:
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain
        
        chain_idx, estimator =self.training_classifier_position, self.classifier
        y = Y[:, self.order_[chain_idx]]
        estimator.fit(X_aug[:, : (X.shape[1] + chain_idx)], y)
        
        
        return (estimator, self.order_, self.training_classifier_position)
    
    def predict_single(self, X,Y):
        """Predicts the labels of the given data but uses the perfect labels instead of the predictions of the previous classifiers.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)        
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")
        
        Y_pred_chain = Y[:, self.order_]
        if sp.issparse(X):
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")
            X_aug = X_aug.tocsr()
        else:
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain
        
        chain_idx, estimator =self.training_classifier_position, self.classifier
        pred=estimator.predict(X_aug[:, : (X.shape[1] + chain_idx)])
        
        return pred


class chaining_together(ClassifierChain,):
    """This class is used to chain together the list of classifiers that were trained separately.
    """
    
    def __init__(self, *, config_classifier=None, order=None):
        """
        A constructor for the class. It initializes the class with the parameters passed to it.
        
        :param config_classifier: The list of trained classifier that will be used to classify the data
        :param order: The order of the classifier
        """
        self.estimators_ = config_classifier
        self.order = order
        self.order_=np.array(order)
        
    def fit(*args, **kwargs):
        raise AssertionError("This method should not be called directly")
    
    def predict_parallel(self, X,cores=1):
        """Parallel prediction of the labels of the given data.
        
        Is slower than the normal predict function, 
        but can be used to speed up the prediction process maybe if the data is large
        """
        X_splits=np.array_split(X,cores)
        pred_lst=Parallel(n_jobs=cores)(delayed(self.predict)(X_split) for X_split in X_splits)
        pred=np.vstack(pred_lst)
        return pred
    
    def predict_single(self, X,pos):
        """
        It takes in a dataframe X, and a position pos, and returns the prediction of the model at that
        position.
        
        :param X: The input data
        :param pos: the position of the estimator in the chain
        :return: The prediction of the model at the position pos.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
            
            if chain_idx == pos:
                Y_pred=Y_pred_chain[:, chain_idx]
                return Y_pred

    def predict_up_to(self, X,pos):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

            if chain_idx == pos:
                Y_pred=Y_pred_chain[:, chain_idx]
                return Y_pred
    

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    import dataset_utils as ds

    X, Y = ds.fetch_openml_dataset("yeast")

    base_lr = LogisticRegression()
    
    single_chain=chain_configuration_single_fit(single_classifier=base_lr,training_classifier_nr=4)
    single_chain.fit(X,Y)
    single_pred=single_chain.predict_single(X,Y)
    print(single_pred.shape)