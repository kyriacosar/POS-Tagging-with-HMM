#!/usr/bin/env python

import nltk, inspect, math, numpy as np

from nltk.corpus import brown
from nltk.tag import map_tag

from nltk.probability import FreqDist

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

from nltk.probability import MLEProbDist
from nltk.probability import LidstoneProbDist

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD: ConditionalProbDist = None
        self.transition_PD: ConditionalProbDist = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """

        # TODO prepare data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        data = []
        for s in train_data:
            temp = [(tag, word.lower()) for (word, tag) in s]
            data.extend(temp)

        # TODO compute the emission model
        emission_FD = ConditionalFreqDist(data)
        self.emission_PD = ConditionalProbDist(emission_FD, LidstoneProbDist, gamma=0.01)
        #self.states = [u'.', u'ADJ', u'ADP', u'ADV', u'CONJ', u'DET', u'NOUN', u'NUM', u'PRON', u'PRT', u'VERB', u'X']
        #self.states = ['NOUN']

        state_list = []
        for w in data:
            state_list.append(w[0])

        self.states = list(set(state_list))

        return self.emission_PD, self.states

    # Compute transition model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        # TODO: prepare the data
        data = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        tags = []
        for s in train_data:
            temp = ['<s>']
            temp_tags = [tag for (word, tag) in s]
            temp.extend(temp_tags)
            temp.append('</s>')
            tags.extend(temp)

        data = [(tags[i], tags[i+1]) for i in range(len(tags) - 1)]

        # TODO compute the transition model

        transition_FD = ConditionalFreqDist(data)
        self.transition_PD = ConditionalProbDist(transition_FD, LidstoneProbDist, gamma=0.01)

        return self.transition_PD

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        # Initialise viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)

        # The viterbi data structure contains the Viterbi path probabilities as a T by N
        # table where T is the number of observations and N is the number of states or tags.
        # Each cell contains the most probable path by taking the maximum over all possible
        # previous state sequences to arrive at that state.
        self.viterbi = []

        # The backpointer data structure keeps track of the best path of hidden states that
        # led to each state in a T by N table where  is the number of observations and N is
        # the number of states. Each cell contains the state which had the maximum viterbi
        # probability (from the Viterbi table) in the previous time step (or observation).
        self.backpointer = []
        self.viterbi.append([])
        self.backpointer.append([])

        for state in self.states:
            self.viterbi[0].extend([- math.log2(self.transition_PD["<s>"].prob(state)) - math.log2(self.emission_PD[state].prob(observation))])
            # Initialise backpointer
            self.backpointer[0].extend(['<s>'])

    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    # Input: list of words
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        tags = []
        index = 0
        current_decision = []

        for t in range(1, len(observations)):
            #print('\nPrevious observation: ', observations[t-1])
            #print('\nCurrent obesrvation: ', observations[t])

            # Appending a new list for storing the current Viterbi costs.
            self.viterbi.append([])

            # Getting the Viterbi costs of the previous observation.
            previous_viterbi = np.array(self.viterbi[t-1])

            # Appending a new list for storing the backpointers for the states of this observation.
            self.backpointer.append([])
            #print('\nPrevious Viterbi column: ')
            #print(previous_viterbi)

            # Double recursion to calculate the best paths for each state from all other states.
            for i in range(len(self.states)):
                current_state = self.states[i]
                #print('Current state: ')
                #print(current_state)

                current_state_transition = []

                for j in range(len(self.states)):
                    previous_state = self.states[j]
                    #print('Previous state: ')
                    #print(previous_state)

                    # Getting the transition cost from the previous state to the current state and storing it.
                    state_transition_cost = -math.log2(self.transition_PD[previous_state].prob(current_state))
                    current_state_transition.append(state_transition_cost)

                # Getting the state observation cost.
                emission_cost = - math.log2(self.emission_PD[current_state].prob(observations[t]))
                #print('Emission cost: ')
                #print(emission_cost)

                # Transforming the current_state_transition list to a numpy array.
                current_state_transition = np.array(current_state_transition)
                #print('Transtions probabilities:')
                #print(current_state_transition)

                # Calculating the Viterbi costs from each previous state to the current state using element-wise addition.
                current_state_costs = previous_viterbi + current_state_transition + emission_cost
                #print('State costs:')
                #print(current_state_costs)

                #print('\nMinimum state: ', self.states[np.argmin(current_state_costs)])

                # Calculating and storing the minimum Viterbi cost for the current state.
                self.viterbi[t].append(np.amin(current_state_costs))

                # Calculating and storing the backpointer for the current state.
                self.backpointer[t].append(np.argmin(current_state_costs))

        # TODO
        # Add cost of termination step (for transition to </s> , end of sentence).

        # Getting the Viterbi costs of the last observation.
        previous_viterbi = np.array(self.viterbi[-1])

        # Appending a new list for storing the Viterbi costs to the end </s> state.
        self.viterbi.append([])

        # Appending a new list for storing the backpointers for the states of the end </s> state.
        self.backpointer.append([])

        current_state_transition = []

        # Recursion over the states of the final observation.
        for i in range(len(self.states)):
            previous_state = self.states[i]

            # Getting and storing the transition cost from the final observation state to the end </s> state.
            state_transition_cost = -math.log2(self.transition_PD[previous_state].prob('</s>'))
            current_state_transition.append(state_transition_cost)

        # Transforming the current_state_transition list to a numpy array.
        current_state_transition = np.array(current_state_transition)

        # Calculating the Viterbi costs from each previous state to the end state using element-wise addition.
        current_state_costs = previous_viterbi + current_state_transition

        # Calculating and storing the minimum Viterbi cost for the end state.
        self.viterbi[len(observations)].append(np.amin(current_state_costs))

        # Calculating and storing the backpointer for the end state.
        self.backpointer[len(observations)].append(np.argmin(current_state_costs))

        #print('\nObservations: ', observations)
        #print('\nViterbi table length: ', len(self.viterbi))
        #print('\nViterbi table:')
        #print(self.viterbi)
        #print('\nBackpointer table length: ', len(self.backpointer))
        #print('\nBackpointer table:')
        #print(self.backpointer)

        # TODO
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.
        tag_index = self.backpointer[len(observations)][0]
        tags.insert(0, self.states[tag_index])

        for i in reversed(range(1, len(observations))):
            tag_index = self.backpointer[i][tag_index]
            tags.insert(0, self.states[tag_index])

        return tags

def answer_question4b():
    """ Report a tagged sequence that is incorrect
    :rtype: str
    :return: your answer [max 280 chars]"""

    tagged_sequence = []
    correct_sequence = []

    correct = 0
    incorrect = 0

    model = HMM(train_data_universal, test_data_universal)
    model.train()

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        count = 0

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                count += 1
            else:
                break

        if count != len(sentence):
            tagged_sequence.append(list(zip(sentence, tags)))
            correct_sequence.append([tag for (word, tag) in sentence])

        if len(tagged_sequence) == 10:
            break

    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""There is an incorrect adjective tag on the word \'Fulton\', while the correct tag is a noun. This is due to the next word being a noun and the probability of a noun following an adjective is higher than a noun following a noun.""")[0:280]

    return tagged_sequence[0], correct_sequence[0], answer

def answer_question5():
    """Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]"""

    return inspect.cleandoc("""\
    fill me in""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # Test of model for sentence with single word 'sentence':
    print('\nSingle word sentence test: ')
    train_test_s = [[('sentence', 'NOUN')]]
    model = HMM(train_test_s, train_test_s)
    model.train()

    s = 'sentence'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print('\nEmission probability of word \'sentence\' being a noun: ', model.emission_PD['NOUN'].prob('sentence'))
    print('Transition probability from start of sentence to noun: ', model.transition_PD['<s>'].prob('NOUN'))
    print('Transition probability from noun of sentence to noun: ', model.transition_PD['NOUN'].prob('NOUN'))
    print('\nTagging of the single word sentence by the model:')
    print(list(zip(s,ttags)))


    # Test of model for sentence with two 'sentence' words:
    print('\n\nDouble word sentence test: ')
    train_test_s = [[('sentence', 'NOUN'), ('sentence', 'NOUN')]]
    model = HMM(train_test_s, train_test_s)
    model.train()

    s = 'sentence sentence'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print('\nEmission probability of word \'sentence\' being a noun: ', model.emission_PD['NOUN'].prob('sentence'))
    print('Transition probability from start of sentence to noun: ', model.transition_PD['<s>'].prob('NOUN'))
    print('Transition probability from noun of sentence to noun: ', model.transition_PD['NOUN'].prob('NOUN'))
    print('\nTagging of the double word sentence by the model:')
    print(list(zip(s,ttags)))

    print('\n\nModel Test on universal tagset: ')
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 1000
    #train_size = len(tagged_sentences_universal) - 1000
    train_size = len(tagged_sentences_universal)

    #test_data_universal = tagged_sentences_universal[(-test_size):]
    #train_data_universal = tagged_sentences_universal[:train_size]

    test_data_universal = tagged_sentences_universal[0:test_size]
    train_data_universal = tagged_sentences_universal[test_size:train_size]

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Inspect the model to see if emission_PD and transition_PD look plausible
    print('\nstates: %s\n'%model.states)

    # Added checks:
    print('Probability of \'are\' being a VERB: ', model.emission_PD['VERB'].prob('are'))
    print('Probability of \'the\' being a DET: ', model.emission_PD['DET'].prob('the'))
    print('Probability of a noun followed by an adjective: ', model.transition_PD['NOUN'].prob('ADJ'))
    print('Probability of an adjective followed by a noun: ', model.transition_PD['ADJ'].prob('NOUN'))

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print('\nTag a trial sentence')
    print(list(zip(s,ttags)))

    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1

    accuracy = correct / (correct + incorrect)
    print('\nTagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b and 5
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nAn incorrect tagged sequence is:')
    print(bad_tags)
    print('\nThe correct tagging of this sentence would be:')
    print(good_tags)
    print('\nA possible reason why this error may have occurred is:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])

if __name__ == '__main__':
    answers()
