# -*- coding: utf-8 -*-

import numpy as np
import h5py

from utils import Utils
import pdb
"""
Tools for steganographic algorithms
"""

class Embedding_simulator():
    @staticmethod
    def binary_entropy(pPM1):
        p0 = 1 - pPM1
        p0[p0==0]=1
        p0[p0 < 0] = 1
        #p0[p0 < np.spacing(1)] = 1
        pPM1[pPM1==0]=1
        P = np.hstack((p0.flatten(1), pPM1.flatten(1)))
        H = -((P) * np.log2(P))
        # Add in order to do the same as matlab
        #H[P < np.spacing(1)] = 0
        #H[P > 1-np.spacing(1)] = 0
        Ht = np.sum(H)
        return Ht

    @staticmethod
    def ternary_entropy(pP1, pM1):
        p0 = 1 - pP1 - pM1
        p0[p0==0]=1
        p0[p0 < 0] = 1
        #p0[p0 < np.spacing(1)] = 1
        pP1[pP1==0]=1
        pM1[pM1==0]=1
        #print("p0: ",p0)
        #P = np.hstack((p0.flatten(1), pP1.flatten(1), pM1.flatten(1)))
        P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
        H = -((P) * np.log2(P))
        # Add in order to do the same as matlab
        #H[P < np.spacing(1)] = 0
        #H[P > 1-np.spacing(1)] = 0
        Ht = np.sum(H)
        return Ht

    @staticmethod
    def calc_lambda_binary(rhoPM1, message_length, n):
        l3 = 1000 # just an initial value
        m3 = message_length + 1 # to enter at least one time in the loop, just an initial value
                                # m3 is the total entropy
        iterations = 0 # iterations counter
        while m3 > message_length:
            """
            This loop returns the biggest l3 such that total entropy (m3) <= message_length
            Stop when total entropy < message_length
            """
            l3 *= 2
            pPM1 = np.exp(-l3 * rhoPM1) / (1 + np.exp(-l3 * rhoPM1))
            # Total entropy
            m3 = Embedding_simulator.binary_entropy(pPM1)
            iterations += 1
            if iterations > 10:
                """
                Probably unbounded => it seems that we can't find beta such that
                message_length will be smaller than requested. Binary search
                doesn't work here
                """
                lbd = l3
                return lbd


        l1 = 0.0 # just an initial value
        m1 = n # just an initial value
        lbd = 0.0

        alpha = message_length / n # embedding rate
        # Limit search to 30 iterations
        # Require that relative payload embedded is roughly within
        # 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000 and iterations < 30:
            lbd = l1 + (l3 - l1) / 2 # dichotomy
            pPM1 = np.exp(-lbd * rhoPM1) / (1 + np.exp(-lbd * rhoPM1))
            m2 = Embedding_simulator.binary_entropy(pPM1) # total entropy new calculation
            if m2 < message_length: # classical binary search
                l3 = lbd
                m3 = m2
            else:
                l1 = lbd
                m1 = m2
            iterations += 1 # for monitoring the number of iterations
        return lbd

    @staticmethod
    def calc_lambda(rhoP1, rhoM1, message_length, n):
        l3 = 1000 # just an initial value
        m3 = message_length + 1 # to enter at least one time in the loop, just an initial value
                                # m3 is the total entropy
        iterations = 0 # iterations counter
        while m3 > message_length:
            """
            This loop returns the biggest l3 such that total entropy (m3) <= message_length
            Stop when total entropy < message_length
            """
            l3 *= 2
            pP1 = np.exp(-l3 * rhoP1) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            pM1 = np.exp(-l3 * rhoM1) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            # Total entropy
            m3 = Embedding_simulator.ternary_entropy(pP1, pM1)
            iterations += 1
            if iterations > 10:
                """
                Probably unbounded => it seems that we can't find beta such that
                message_length will be smaller than requested. Ternary search 
                doesn't work here
                """
                lbd = l3
                return lbd


        l1 = 0.0 # just an initial value
        m1 = n # just an initial value
        lbd = 0.0

        alpha = message_length / n # embedding rate
        # Limit search to 30 iterations
        # Require that relative payload embedded is roughly within
        # 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000 and iterations < 30:
            lbd = l1 + (l3 - l1) / 2 # dichotomy
            pP1 = np.exp(-lbd * rhoP1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
            pM1 = np.exp(-lbd * rhoM1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))

            m2 = Embedding_simulator.ternary_entropy(pP1, pM1) # total entropy new calculation
            if m2 < message_length: # classical ternary search
                l3 = lbd
                m3 = m2
            else:
                l1 = lbd
                m1 = m2
            iterations += 1 # for monitoring the number of iterations
        return lbd

    @staticmethod
    def calc_lambda_0(rhoP1, rhoM1, rho0, message_length, n):
        l3 = 10 # just an initial value
        m3 = message_length + 1 # to enter at least one time in the loop, just an initial value
                                # m3 is the total entropy
        iterations = 0 # iterations counter
        while m3 > message_length:
            """
            This loop returns the biggest l3 such that total entropy (m3) <= message_length
            Stop when total entropy < message_length
            """
            l3 *= 2
            # pP1 = np.exp(-l3 * rhoP1) / (np.exp(-l3 * rho0) + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            # pM1 = np.exp(-l3 * rhoM1) / (np.exp(-l3 * rho0) + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            pP1 = 1 / (1 + np.exp(-l3 * (rhoM1-rhoP1)) + np.exp(-l3* (rho0 - rhoP1)))
            pM1 = 1 / (1 + np.exp(-l3 * (rhoP1 - rhoM1)) + np.exp(-l3 * (rho0 - rhoM1)))
            # Total entropy
            m3 = Embedding_simulator.ternary_entropy(pP1, pM1)
            iterations += 1
            if iterations > 10:
                """
                Probably unbounded => it seems that we can't find beta such that
                message_length will be smaller than requested. Ternary search 
                doesn't work here
                """
                lbd = l3
                return lbd


        l1 = 0.0 # just an initial value
        m1 = n # just an initial value
        lbd = 0.0

        alpha = message_length / n # embedding rate
        # Limit search to 30 iterations
        # Require that relative payload embedded is roughly within
        # 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000 and iterations < 50:
            lbd = l1 + (l3 - l1) / 2 # dichotomy
            # pP1 = np.exp(-lbd * rhoP1) / (np.exp(-l3 * rho0) + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
            pP1 = 1 / (1 + np.exp(-lbd * (rhoM1-rhoP1)) + np.exp(-lbd * (rho0 - rhoP1)))
            pM1 = 1 / (1 + np.exp(-lbd * (rhoP1 - rhoM1)) + np.exp(-lbd * (rho0 - rhoM1)))
            # pdb.set_trace()
            m2 = Embedding_simulator.ternary_entropy(pP1, pM1) # total entropy new calculation
            if m2 < message_length: # classical ternary search
                l3 = lbd
                m3 = m2
            else:
                l1 = lbd
                m1 = m2
            iterations += 1 # for monitoring the number of iterations
        return lbd

    @staticmethod
    def compute_proba_binary(rhoPM1, message_length, n):
        """
        Embedding simulator simulates the embedding made by the best possible
        binary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """
        lbd = Embedding_simulator.calc_lambda_binary(rhoPM1, message_length, n)
        p_change_PM1 = np.exp(-lbd * rhoPM1) / (1 + np.exp(-lbd * rhoPM1))
        return p_change_PM1

    @staticmethod
    def compute_proba(rhoP1, rhoM1, message_length, n):
        """
        Embedding simulator simulates the embedding made by the best possible 
        ternary coding method (it embeds on the entropy bound). This can be 
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC) 
        that are asymptotically approaching the bound
        """
        lbd = Embedding_simulator.calc_lambda(rhoP1, rhoM1, message_length, n)
        p_change_P1 = np.exp(-lbd * rhoP1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        p_change_M1 = np.exp(-lbd * rhoM1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        return p_change_P1, p_change_M1

    @staticmethod
    def compute_proba_0(rhoP1, rhoM1, rho0, message_length, n):
        """
        Embedding simulator simulates the embedding made by the best possible 
        ternary coding method (it embeds on the entropy bound). This can be 
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC) 
        that are asymptotically approaching the bound
        The cost of not modifying the quantized coefficient is not 0
        """
        lbd = Embedding_simulator.calc_lambda_0(rhoP1, rhoM1, rho0, message_length, n)
        # p_change_P1 = np.exp(-lbd * rhoP1) / (np.exp(-lbd * rho0) + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        # p_change_M1 = np.exp(-lbd * rhoM1) / (np.exp(-lbd * rho0) + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        p_change_P1 = 1 / (1 + np.exp(-lbd * (rhoM1-rhoP1)) + np.exp(-lbd * (rho0 - rhoP1)))
        p_change_M1 = 1 / (1 + np.exp(-lbd * (rhoP1 - rhoM1)) + np.exp(-lbd * (rho0 - rhoM1)))
        return p_change_P1, p_change_M1

    @staticmethod
    def process_binary(cover, rhoPM1, message_length):
        """
        Embedding simulator simulates the embedding made by the best possible
        binary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """

        Utils.randseed()

        n = cover.size
        p_change_PM1 = Embedding_simulator.compute_proba_binary(rhoPM1, message_length, n)
        # XXX: to change
        randChange = Utils.randsample((cover.shape[0], cover.shape[1]))
        y = np.copy(cover)
        y[randChange < p_change_PM1] = y[randChange < p_change_PM1] - 1 + 2 * Utils.randint(2)
        return y, p_change_PM1

    @staticmethod
    def process(cover, p_change_P1, p_change_M1):
        """
        Embedding simulator simulates the embedding made by the best possible 
        ternary coding method (it embeds on the entropy bound). This can be 
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC) 
        that are asymptotically approaching the bound
        """

        Utils.randseed()

        n = cover.size
        # XXX: to change
        if cover.ndim>2:
            randChange = Utils.randsample((cover.shape[0], cover.shape[1]))
        else:
            randChange = Utils.randsample(cover.shape)
        
        y = np.copy(cover)
        y[randChange < p_change_P1] = y[randChange < p_change_P1] + 1
        y[np.logical_and(randChange >= p_change_P1, randChange < (p_change_P1 + p_change_M1))] = y[np.logical_and(randChange >= p_change_P1, randChange < (p_change_P1 + p_change_M1))] - 1
        return y

    @staticmethod
    def create_hdf5(filename, buffer):
        p = h5py.File(filename,'w')
        p.create_dataset('dataset_1', data=buffer)
        p.close()
