from unittest import result
import math

from Validators.keypoint_validator import CubicSplineKeyPointInterpolator


class AnomalyDetection():
    def __init__(self, validator: CubicSplineKeyPointInterpolator):
        self.validator = validator
        self.clusters = self.validator.findNonMissingValueClusters()
        self.movement_flags = None
        self.posision_flags = self.validator.flagAbnormalDistances()
    
    def movementAnomalies(self, 
                         threshold=0.15):
        '''returns hands with flagged speeds'''
        if threshold:
            return self.validator.findMovmentClusters(max_momentum=threshold)
        elif self.movement_flags == None:
            self.movement_flags = self.validator.findMovmentClusters()
        return self.movement_flags
    
    def positionAnomalies(self, 
                         threshold=-0.15):
        '''returns hands with flagged positions (too far on the wrong side)'''
        if threshold:
            return self.validator.flagAbnormalDistances(outlier_boundry=threshold)
        elif self.posision_flags == None:
            self.posision_flags = self.validator.flagAbnormalDistances() 
        return self.posision_flags
    
    def filledMovementAnomalies(self,
                                gap_size=5,
                                threshold=0.15):
        flags = self.movementAnomalies(threshold=threshold)
        
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
    
    def filledMovementAnomaliesByMAD(self,
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
    
    def filledMovementAnomaliesByPercentile(self, percentile, gap_size=5):
        
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
    
    def filledMovementAnomaliesByStdDev(self, num_std_dev, gap_size=5):
        
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
                    # too ambiguous to include then drops it
                    run_start = i + 1
            
            result.append(trimmed)
        
        return result[0], result[1]

    def positionAndMovementAnomalies(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15):
        movement = self.movementAnomalies(threshold=movement_threshole)
        posision = self.positionAnomalies(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        
        return list(sorted(left)), list(sorted(right))
    
    def positionAndFilledMovementAnomalies(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15,
                                   gap_size=5):
        movement = self.filledMovementAnomalies(gap_size=gap_size, 
                                               threshold=movement_threshole)
        posision = self.positionAnomalies(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def posisionAndFilledMovmentAnomalysByStdDev(self, 
                                   position_threshold=-0.15, 
                                   movement_threshole=0.15,
                                   gap_size=5,
                                   num_std_dev=1.5):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                       gap_size=gap_size)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def posisionAndFilledMovmentAnomalysByMAD(self,
                                   position_threshold=-0.15,
                                   movement_threshole=0.15,
                                   threshold=1.0,
                                   gap_size=5):
        movement = self.filledMovementAnomalysByMAD(threshold=threshold, 
                                                    gap_size=gap_size)
        posision = self.posisionAnomalys(threshold=position_threshold)
        
        left =set(movement[0]) | set(posision[0])
        
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def posisionAndFilledMovmentAnomalysByPercentile(self,
                                      position_threshold=-0.15,
                                      movement_threshole=0.15,
                                      percentile=95,
                                      gap_size=5):
          movement = self.filledMovementAnomalysByPercentile(percentile=percentile, 
                                                             gap_size=gap_size)
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
    
    def handOrderingAnomalysByPalmCenterUsingNeighbourFilling(self, 
                                                              margin=0.0):
        """Detects hand ordering violations using palm center X comparison
        with neighbour-filled positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByPalmCenterUsingNeighbourFilling(margin=margin)
        return violations, violations

    def handOrderingAnomalysByWristUsingNeighbourFilling(self, 
                                                         margin=0.0):
        """Detects hand ordering violations using wrist X comparison
        with neighbour-filled positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByWristUsingNeighbourFilling(
            margin=margin)
        return violations, violations

    def handOrderingAnomalysByExtremesUsingNeighbourFilling(self, 
                                                            margin=0.0):
        """Detects hand ordering violations using extreme keypoint X comparison
        with neighbour-filled positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByExtremesUsingNeighbourFilling(
            margin=margin)
        return violations, violations

    def handOrderingAnomalysByPalmCenterUsingInterpolation(self, 
                                                           margin=0.0):
        """Detects hand ordering violations using palm center X comparison
        with interpolated positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByPalmCenterUsingInterpolation(
            margin=margin)
        return violations, violations

    def handOrderingAnomalysByWristUsingInterpolation(self, 
                                                      margin=0.0):
        """Detects hand ordering violations using wrist X comparison
        with interpolated positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByWristUsingInterpolation(
            margin=margin)
        return violations, violations

    def handOrderingAnomalysByExtremesUsingInterpolation(self, 
                                                         margin=0.0):
        """Detects hand ordering violations using extreme keypoint X comparison
        with interpolated positions for missing hands.
        
        takes:
            margin: minimum x-distance to count as a violation
        returns:
            tuple of (frame_indices, frame_indices) — violations affect both hands equally
        """
        violations = self.validator.findHandOrderingByExtremesUsingInterpolation(
            margin=margin)
        return violations, violations
    
    def OrderingByPalmsWithNeighbourFillingAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        
        posision = self.handOrderingAnomalysByPalmCenterUsingNeighbourFilling(margin=margin)
        
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByWristsWithNeighbourFillingAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByWristUsingNeighbourFilling(margin=margin)
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByExtremesWithNeighbourFillingAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByExtremesUsingNeighbourFilling(margin=margin)
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByPalmsWithInterpolationAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByPalmCenterUsingInterpolation(margin=margin)
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))

    def OrderingByWristsWithInterpolationAndFilledMovmentByStdDevIntersection(self,
                                                    position_threshold=-0.15, 
                                                    num_std_dev=2.0,
                                                    gap_size=5,
                                                    margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByWristUsingInterpolation(margin=margin)
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByExtremesWithInterpolationAndFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByExtremesUsingInterpolation(margin=margin)
        left = set(movement[0]) & set(posision[0])
        right = set(movement[1]) & set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByPalmsWithNeighbourFillingOrFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        
        posision = self.handOrderingAnomalysByPalmCenterUsingNeighbourFilling(margin=margin)
        
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByWristsWithNeighbourFillingOrFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByWristUsingNeighbourFilling(margin=margin)
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByExtremesWithNeighbourFillingOrFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByExtremesUsingNeighbourFilling(margin=margin)
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByPalmsWithInterpolationOrFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByPalmCenterUsingInterpolation(margin=margin)
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))

    def OrderingByWristsWithInterpolationOrFilledMovmentByStdDevIntersection(self,
                                                    position_threshold=-0.15, 
                                                    num_std_dev=2.0,
                                                    gap_size=5,
                                                    margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByWristUsingInterpolation(margin=margin)
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def OrderingByExtremesWithInterpolationOrFilledMovmentByStdDevIntersection(self, 
                                                   position_threshold=-0.15, 
                                                   num_std_dev=2.0,
                                                   gap_size=5,
                                                   margin=0.0):
        movement = self.filledMovementAnomalysByStdDev(num_std_dev=num_std_dev, 
                                                        gap_size=gap_size)
        posision = self.handOrderingAnomalysByExtremesUsingInterpolation(margin=margin)
        left = set(movement[0]) | set(posision[0])
        right = set(movement[1]) | set(posision[1])
        return list(sorted(left)), list(sorted(right))
    
    def AccelerationAnomalys(self, threshold=0.15, inclusive=True, interpolate_missing=False):
        result = self.validator.findAccelerationClusters(margin=threshold, 
                                                       inclusive=inclusive, 
                                                       interpolate_missing=interpolate_missing)
        # Ensure result is a tuple (left_flags, right_flags)
        if isinstance(result, dict):
            return result.get('left', []), result.get('right', [])
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            return result[0], result[1]
        else:
            return result, result
    
    def positionAndFilledMovementAndAccelerationAnomalies(self, 
                                               position_threshold=-0.1, 
                                               movement_threshold=0.1, 
                                               acceleration_threshold=0.1,
                                               gap_size=5,
                                               inclusive=True, 
                                               interpolate_missing=False
                                               ):
        movement = self.filledMovementAnomalys(gap_size=gap_size, threshold=movement_threshold)
        acceleration = self.AccelerationAnomalys(threshold=acceleration_threshold, 
                                                 inclusive=inclusive, 
                                                 interpolate_missing=interpolate_missing)
        position = self.posisionAnomalys(threshold=position_threshold)
        
        left = set(movement[0]) | set(position[0]) | set(acceleration[0])
        right = set(movement[1]) | set(position[1]) | set(acceleration[1])
        
        return list(sorted(left)), list(sorted(right))
    
    def findAppearanceDisappearanceSwaps(self, max_gap=2, distance_threshold=0.15, show_logs=False):
        """Detects frames where hand data jumps between slots.
        
        Flags frame t if:
        - Slot A was empty at t-1 (or within max_gap frames) and has data at t
        - Slot B had data recently and is empty at t
        - The position in slot A at frame t is close to where slot B's 
            data was when it was last seen
        
        takes:
            max_gap: how many frames back to look for the disappearance
            distance_threshold: max distance between appearing hand and 
                            last known position of disappearing hand
        returns:
            list of frame indices where appearance-disappearance swap detected
        """
        if self.validator._KeyPointValidator__palms is None:
            self.validator.findAllPalmCenters()
        
        palms = self.validator._KeyPointValidator__palms
        violations = []
        
        for i in range(1, len(palms)):
            for appearing_side, disappearing_side in [('left', 'right'), ('right', 'left')]:
                # Check: appearing_side has data now
                current_appearing = palms[i].get(appearing_side)
                if (not current_appearing or current_appearing == [None, None] 
                    or None in current_appearing):
                    continue
                
                # Check: appearing_side was empty recently
                was_empty = True
                for back in range(1, min(max_gap + 1, i + 1)):
                    prev = palms[i - back].get(appearing_side)
                    if prev and prev != [None, None] and None not in prev:
                        was_empty = False
                        break
                if not was_empty:
                    continue
                
                # Check: disappearing_side had data recently and is now empty
                current_disappearing = palms[i].get(disappearing_side)
                is_now_empty = (not current_disappearing 
                            or current_disappearing == [None, None] 
                            or None in current_disappearing)
                
                if not is_now_empty:
                    continue  # both hands present — not a single-hand swap
                
                # Find the last known position of the disappearing hand
                last_known_pos = None
                for back in range(1, min(max_gap + 2, i + 1)):
                    prev = palms[i - back].get(disappearing_side)
                    if prev and prev != [None, None] and None not in prev:
                        last_known_pos = prev
                        break
                
                if last_known_pos is None:
                    continue  # disappearing hand was never seen
                
                # Check: appearing hand is close to where disappearing hand was
                dist = math.hypot(
                    current_appearing[0] - last_known_pos[0],
                    current_appearing[1] - last_known_pos[1]
                )
                
                if dist < distance_threshold:
                    violations.append(i)
                    if show_logs:
                        print(f"Frame {i}: {appearing_side} appeared at "
                            f"({current_appearing[0]:.3f}, {current_appearing[1]:.3f}), "
                            f"{disappearing_side} disappeared from "
                            f"({last_known_pos[0]:.3f}, {last_known_pos[1]:.3f}), "
                            f"dist={dist:.4f}")
        
        # Return as tuple (left_flags, right_flags) — violations affect both hands equally
        return violations, violations