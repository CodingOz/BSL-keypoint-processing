
import os
import random

# aims to help collect a random but representative corpus of 33 valid sign
# keypoints


class SubcorpusSelector:
    def __init__(self, corpus):
        self.corpus = corpus

    def listSigns(self, sign):
        sign = sign.upper()
        # returns all submitons in the folder: corpus-path/sign
        path = os.path.join(self.corpus, sign)
        return [
            f for f in os.listdir(path) if os.path.isfile(
                os.path.join(
                    path, f))]

    def selectRandomSigns(self, sign, n=33):
        sign = sign.upper()
        submissions = self.listSigns(sign)
        if len(submissions) < n:
            print(
                f"Warning: Only {len(submissions)} submissions available for sign '{sign}'. Returning all.")
            return submissions
        return random.sample(submissions, n)

    def extractSubmissionId(self, filename):
        # Extract submission ID from filename (e.g.,
        # "1be98b34-0edc-41ee-871c-e592c0b4198f.json" ->
        # "1be98b34-0edc-41ee-871c-e592c0b4198f")
        return filename[:-5] if filename.endswith('.json') else filename

    def checkDuplicateSubmissions(self, selected_dict):
        # Checks if there are any duplicate submission IDs across all selected signs
        # Returns a dict with duplicate IDs as keys and list of signs they
        # appear in as values
        submission_to_signs = {}

        for sign, submissions in selected_dict.items():
            for submission_file in submissions:
                sub_id = self.extractSubmissionId(submission_file)
                if sub_id not in submission_to_signs:
                    submission_to_signs[sub_id] = []
                submission_to_signs[sub_id].append(sign)

        duplicates = {sub_id: signs for sub_id,
                      signs in submission_to_signs.items() if len(signs) > 1}
        return duplicates

    def selectRandomSignsFromAll(self, signs=[], n_submissions=3):
        # ensures that no 2 signs are from the same submission, and that we get a good variety of signs
        # processes signs in order from lowest to highest number of submissions
        # signs: list of signs to select from (if empty, selects from all signs in corpus)
        # n_submissions: number of submissions to select per sign
        all_signs = [
            d for d in os.listdir(
                self.corpus) if os.path.isdir(
                os.path.join(
                    self.corpus,
                    d))]
        if signs:
            selected = [sign for sign in all_signs if sign in signs]
        else:
            selected = random.sample(
                all_signs, min(
                    n_submissions, len(all_signs)))

        # sort selected signs by number of submissions (ascending)
        sign_submission_counts = [
            (sign, len(self.listSigns(sign))) for sign in selected]
        sign_submission_counts.sort(key=lambda x: x[1])
        sorted_signs = [sign for sign, count in sign_submission_counts]

        self.selected_signs = sorted_signs

        result = {}
        used_submission_ids = set()

        for sign in sorted_signs:
            submissions = self.listSigns(sign)
            # Extract submission IDs from filenames
            submissions_with_ids = [(f[:-5], f)
                                    for f in submissions if f.endswith('.json')]

            # shuffle to get random selections
            random.shuffle(submissions_with_ids)

            selected_for_sign = []
            for submission_id, filename in submissions_with_ids:
                if submission_id not in used_submission_ids:
                    selected_for_sign.append(filename)
                    used_submission_ids.add(submission_id)
                    if len(selected_for_sign) >= n_submissions:
                        break

            if len(selected_for_sign) < n_submissions:
                print(
                    f"Warning: Only {len(selected_for_sign)} unique submissions available for sign '{sign}' without duplicates.")

            result[sign] = selected_for_sign

        self.selected_signs = result
        return result

    def reselectSigns(self, signId):
        # replaces a specific id if its found to be invalid
        for sign, submissions in self.selected_signs.items():
            if signId in submissions:
                submissions.remove(signId)
                new_selection = self.selectRandomSigns(sign, n=1)
                if new_selection:
                    submissions.append(new_selection[0])
                else:
                    print(
                        f"No more valid submissions available for sign '{sign}' to replace '{signId}'.")
                break
        return new_selection


selector = SubcorpusSelector(
    r"C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed")

# letter signs only
print(
    selector.selectRandomSignsFromAll(
        signs=[
            'A',
            'E',
            'I',
            'O',
            'U',
            'B',
            'S',
            'N',
            'P',
            'T',
            'J'],
        n_submissions=3))
