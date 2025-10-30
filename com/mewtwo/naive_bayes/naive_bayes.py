"""A simple implementation of Naive Bayes classifier for email spam detection."""
import numpy as np
import pandas as pd

emails = pd.read_csv("emails.csv")

def process_email(text):
    """Data preprocessing: turning email text string into a list of words"""
    text = text.lower()
    return list(set(text.split()))

emails['words'] = emails['text'].apply(process_email)

spam_percentage = sum(emails['spam'])/len(emails)

model = {}

for index, email in emails.iterrows():
    for word in email['words']:
        if word not in model:
            model[word] = {'spam': 1, 'ham': 1}
        if word in model:
            if email['spam']:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

def predict_bayes(singular_word):
    """predict the possibility that an email is spam given it contains the singular_word"""
    singular_word = singular_word.lower()
    num_spam_with_word = model[singular_word]['spam']
    num_ham_with_word = model[singular_word]['ham']
    return 1.0*num_spam_with_word/(num_spam_with_word + num_ham_with_word)
#
# print(predict_bayes('lottery'))
# print(predict_bayes('sale'))

def predict_naive_bayes(words_input):
    """predict the possibility of a (set of) keywords in email's appearance indicating spam"""
    ##calculate the total email model figures
    total = len(emails)
    num_spam = sum(emails['spam'])
    num_ham = total - num_spam
    spams = [1.0]
    hams = [1.0]

    ##process the mail input (split into single words in lowercase)
    words_input = words_input.lower()
    words_set = set(words_input.split())

    for word in words_set:
        if word in model:
            spams.append(model[word]['spam']/num_spam*total)
            hams.append(model[word]['ham']/num_ham*total)

    prod_spams = np.long(np.prod(spams)*num_spam)
    prod_hams = np.long(np.prod(hams)*num_ham)

    return prod_spams/(prod_spams + prod_hams)

print(predict_naive_bayes('lottery sale'))
