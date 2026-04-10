from unittest import result

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
        flags = self.movementAnomalys(threshold=threshold)
        
        # get interpolated frame indices so we don't flag estimated values
        _, estimation_flags = self.validator.getFilledPalmCenters()
        interpolated = {
            'left':  {i for i, f in enumerate(estimation_flags) if f.get('left')},
            'right': {i for i, f in enumerate(estimation_flags) if f.get('right')},
        }
        
        sides = ('left', 'right')
        result = []
        
        for indx, side_flags in enumerate(flags):
            side = sides[indx]
            
            # remove interpolated frames from raw flags before reasoning
            clean_flags = sorted(f for f in side_flags if f not in interpolated[side])
            
            if not clean_flags:
                result.append([])
                continue
            
            # gap-fill: build contiguous runs by merging close flags
            filled = list(clean_flags)
            for i in range(len(clean_flags) - 1):
                current = clean_flags[i]
                nxt     = clean_flags[i + 1]
                if (nxt - current) <= gap_size:
                    for j in range(current + 1, nxt):
                        filled.append(j)
            filled = sorted(set(filled))
            
            # trim outer frames of each contiguous run — those are clean boundary frames
            trimmed = []
            run_start = 0
            for i in range(len(filled)):
                is_last = (i == len(filled) - 1)
                end_of_run = is_last or (filled[i + 1] - filled[i] > 1)
                
                if end_of_run:
                    run = filled[run_start:i + 1]
                    if len(run) >= 3:
                        trimmed.extend(run[1:-1])   # drop boundary frames
                    elif len(run) == 2:
                        # two flags = one spike, genuinely ambiguous
                        # keep both rather than discarding everything
                        trimmed.extend(run)
                    # single frame run: isolated spike with no gap-fill neighbour
                    # too ambiguous to include — drop it
                    run_start = i + 1
            
            result.append(trimmed)
        
        return result[0], result[1]
    
    def filledMovementAnomalysByMAD(self,
                                gap_size=5,
                                threshold=0.15):
        flags = self.validator.findMovementRelativeByMAD(threshold=threshold)
        
        # get interpolated frame indices so we don't flag estimated values
        _, estimation_flags = self.validator.getFilledPalmCenters()
        interpolated = {
            'left':  {i for i, f in enumerate(estimation_flags) if f.get('left')},
            'right': {i for i, f in enumerate(estimation_flags) if f.get('right')},
        }
        
        sides = ('left', 'right')
        result = []
        
        for indx, side_flags in enumerate(flags):
            side = sides[indx]
            
            # remove interpolated frames from raw flags before reasoning
            clean_flags = sorted(f for f in side_flags if f not in interpolated[side])
            
            if not clean_flags:
                result.append([])
                continue
            
            # gap-fill: build contiguous runs by merging close flags
            filled = list(clean_flags)
            for i in range(len(clean_flags) - 1):
                current = clean_flags[i]
                nxt     = clean_flags[i + 1]
                if (nxt - current) <= gap_size:
                    for j in range(current + 1, nxt):
                        filled.append(j)
            filled = sorted(set(filled))
            
            # trim outer frames of each contiguous run — those are clean boundary frames
            trimmed = []
            run_start = 0
            for i in range(len(filled)):
                is_last = (i == len(filled) - 1)
                end_of_run = is_last or (filled[i + 1] - filled[i] > 1)
                
                if end_of_run:
                    run = filled[run_start:i + 1]
                    if len(run) >= 3:
                        trimmed.extend(run[1:-1])   # drop boundary frames
                    elif len(run) == 2:
                        # two flags = one spike, genuinely ambiguous
                        # keep both rather than discarding everything
                        trimmed.extend(run)
                    # single frame run: isolated spike with no gap-fill neighbour
                    # too ambiguous to include — drop it
                    run_start = i + 1
            
            result.append(trimmed)
        
        return result[0], result[1]
    
    def filledMovementAnomalysByPercentile(self, percentile, gap_size=5):
        
        flags = self.validator.findMovementRelativeByPercentile(percentile=percentile)
        
        # get interpolated frame indices so we don't flag estimated values
        _, estimation_flags = self.validator.getFilledPalmCenters()
        interpolated = {
            'left':  {i for i, f in enumerate(estimation_flags) if f.get('left')},
            'right': {i for i, f in enumerate(estimation_flags) if f.get('right')},
        }
        
        sides = ('left', 'right')
        result = []
        
        for indx, side_flags in enumerate(flags):
            side = sides[indx]
            
            # remove interpolated frames from raw flags before reasoning
            clean_flags = sorted(f for f in side_flags if f not in interpolated[side])
            
            if not clean_flags:
                result.append([])
                continue
            
            # gap-fill: build contiguous runs by merging close flags
            filled = list(clean_flags)
            for i in range(len(clean_flags) - 1):
                current = clean_flags[i]
                nxt     = clean_flags[i + 1]
                if (nxt - current) <= gap_size:
                    for j in range(current + 1, nxt):
                        filled.append(j)
            filled = sorted(set(filled))
            
            # trim outer frames of each contiguous run — those are clean boundary frames
            trimmed = []
            run_start = 0
            for i in range(len(filled)):
                is_last = (i == len(filled) - 1)
                end_of_run = is_last or (filled[i + 1] - filled[i] > 1)
                
                if end_of_run:
                    run = filled[run_start:i + 1]
                    if len(run) >= 3:
                        trimmed.extend(run[1:-1])   # drop boundary frames
                    elif len(run) == 2:
                        # two flags = one spike, genuinely ambiguous
                        # keep both rather than discarding everything
                        trimmed.extend(run)
                    # single frame run: isolated spike with no gap-fill neighbour
                    # too ambiguous to include — drop it
                    run_start = i + 1
            
            result.append(trimmed)
        
        return result[0], result[1]
    
    def filledMovementAnomalysByStdDev(self, num_std_dev, gap_size=5):
        
        flags = self.validator.findMovementRelativeByStdDev(num_std_dev=num_std_dev)
        
        # get interpolated frame indices so we don't flag estimated values
        _, estimation_flags = self.validator.getFilledPalmCenters()
        interpolated = {
            'left':  {i for i, f in enumerate(estimation_flags) if f.get('left')},
            'right': {i for i, f in enumerate(estimation_flags) if f.get('right')},
        }
        
        sides = ('left', 'right')
        result = []
        
        for indx, side_flags in enumerate(flags):
            side = sides[indx]
            
            # remove interpolated frames from raw flags before reasoning
            clean_flags = sorted(f for f in side_flags if f not in interpolated[side])
            
            if not clean_flags:
                result.append([])
                continue
            
            # gap-fill: build contiguous runs by merging close flags
            filled = list(clean_flags)
            for i in range(len(clean_flags) - 1):
                current = clean_flags[i]
                nxt     = clean_flags[i + 1]
                if (nxt - current) <= gap_size:
                    for j in range(current + 1, nxt):
                        filled.append(j)
            filled = sorted(set(filled))
            
            # trim outer frames of each contiguous run — those are clean boundary frames
            trimmed = []
            run_start = 0
            for i in range(len(filled)):
                is_last = (i == len(filled) - 1)
                end_of_run = is_last or (filled[i + 1] - filled[i] > 1)
                
                if end_of_run:
                    run = filled[run_start:i + 1]
                    if len(run) >= 3:
                        trimmed.extend(run[1:-1])   # drop boundary frames
                    elif len(run) == 2:
                        # two flags = one spike, genuinely ambiguous
                        # keep both rather than discarding everything
                        trimmed.extend(run)
                    # single frame run: isolated spike with no gap-fill neighbour
                    # too ambiguous to include — drop it
                    run_start = i + 1
            
            result.append(trimmed)
        
        return result[0], result[1]

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
    
    def posisionAndFilledMovmentAnomalysByStdDev(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15,
                                   gap_size=5,
                                   num_std_dev=1.5):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, gap_size=gap_size)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def posisionAndFilledMovmentAnomalysByMAD(self,
                                   position_threshold=-0.15,
                                   movement_threshole=0.15,
                                   threshold=1.0,
                                   gap_size=5):
        movement = self.filledMovementAnomalysByMAD(threshold=threshold, gap_size=gap_size)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def posisionAndFilledMovmentAnomalysByPercentile(self,
                                      position_threshold=-0.15,
                                      movement_threshole=0.15,
                                      percentile=95,
                                      gap_size=5):
          movement = self.filledMovementAnomalysByPercentile(percentile=percentile, gap_size=gap_size)
          posision = self.posisionAnomalys(threshold=position_threshold)
          
          left =set(movement[0]) | set(posision[0])
          
          right = set(movement[1]) | set(posision[1])
          return list(sorted(left)), list(sorted(right))
      
    def posisionAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
                                                     