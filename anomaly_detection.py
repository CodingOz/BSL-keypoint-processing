from Validators.keypoint_validator import CubicSplineKeyPointInterpolator

class AnomalyDetection():
    def __init__(self, validator: CubicSplineKeyPointInterpolator):
        self.validator = validator
        self.clusters = self.validator.findNonMissingValueClusters()
        self.movement_flags = None
        self.posision_flags = self.validator.flagAbnormalDistances()
    
    def movementAnomalys(self, 
                         threshold=0.15):
        '''returns hands with flagged speeds'''
        if threshold:
            return self.validator.findMovmentClusters(max_momentum=threshold)
        elif self.movement_flags == None:
            self.movement_flags = self.validator.findMovmentClusters()
        return self.movement_flags
    
    def posisionAnomalys(self, 
                         threshold=-0.15):
        '''returns hands with flagged positions (too far on the wrong side)'''
        if threshold:
            return self.validator.flagAbnormalDistances(outlier_boundry=threshold)
        elif self.posision_flags == None:
            self.posision_flags = self.validator.flagAbnormalDistances() 
        return self.posision_flags
    
    def filledMovementAnomalys(self, 
                               gap_size=5,
                               threshold=0.15):
        '''assumes that frames surounded by movement anomalys may also be anomalous
        takes:
            gap_size: the max space between anomalys where it can be assumed the space is also anomalous
            threshold: the max_momentum to be used
        returns: 
            filled_movement_flags: the filled in movment flags
        '''
        flags = self.movementAnomalys(threshold=threshold)
        for indx, side in enumerate(flags):
            for i in range(len(side)-1):
                current = side[i]
                next = side[i+1]
                if (next-current) <= gap_size:
                    for j in range(current+1,next):
                        flags[indx].append(j)
        return sorted(flags[0]), sorted(flags[1])
    
    def posisionAndMovmentAnomalys(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15):
        movement = self.movementAnomalys(threshold=movement_threshole)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    
    def posisionAndFilledMovmentAnomalys(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15,
                                   gap_size=5):
        movement = self.filledMovementAnomalys(gap_size=gap_size, 
                                               threshold=movement_threshole)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))