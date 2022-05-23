from models.NaiveBayes import NaiveBayes
from utils.load_data import load_data


def main():
    train_data, test_data, train_label, test_label = load_data('data/SMSSpamCollection.txt')
    model = NaiveBayes()
    model.fit(train_data, train_label)
    print(model.predict(test_data))


if __name__ == '__main__':
    main()
