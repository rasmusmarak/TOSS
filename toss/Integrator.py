# General
from enum import Enum
 
class IntegrationScheme(Enum):
        """
         Describes different types of numerical integrators provided by Desolver.
        """
        
        ## Explicit integrators: ##

        # Adaptive methods:
        RK1412Solver = 1
        RK108Solver = 2
        RK8713MSolver = 3
        RK45CKSolver = 4
        HeunEulerSolver = 5
        DOPRI45 = 6

        # Fixed step methods:
        BABs9o7HSolver = 7
        ABAs5o6HSolver = 8
        RK5Solver = 9
        RK4Solver = 10
        MidpointSolver = 11
        HeunsSolver = 12
        EulerSolver = 13
        EulerTrapSolver = 14
        SymplecticEulerSolver = 15
        


        ## Implicit integrators: ##
        
        # Adaptive methods
        LobattoIIIC4 = 16
        RadauIIA5 = 17

        # Fixed step methods
        GaussLegendre4 = 18
        GaussLegendre6 = 19
        BackwardEuler = 20
        ImplicitMidpoint = 21
        LobattoIIIA2 = 22
        LobattoIIIA4 = 23
        LobattoIIIB2 = 24
        LobattoIIIB4 = 25
        LobattoIIIC2 = 26
        CrankNicolson = 27
        RadauIA3 = 28
        RadauIA5 = 29
        RadauIIA3 = 30
        RadauIIA19 = 31