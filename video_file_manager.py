from cryptography.fernet import Fernet
import logging
from botocore.session import Session
import boto3
import os
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import shutil


class EncryptionManager:
    ''' Handles encryption and decryption of video files using 
    Fernet symmetric encryption to stay in line with data protection standards.
    Methods:
        generate_encryption_key: Generates and saves a new encryption key
        get_encryption_key: Retrieves the encryption key from file
        encrypt_file: Encrypts a file using the encryption key
        decrypt_file: Decrypts a file using the encryption key
    '''
    def generate_encryption_key(self):
        try:
            key = Fernet.generate_key()
            key_location = os.getenv('PATH_TO_KEY')
            os.makedirs(key_location, exist_ok=True)
            with open(os.path.join(key_location, 'encryption_key.key'), 'wb') as key_file:
                key_file.write(key)
            return key
        except Exception as e:
            logging.error(f"Failed to generate encryption key: {e}")
            return None
    
    def get_encryption_key(self):
        try:
            load_dotenv()
            key_location = os.getenv('PATH_TO_KEY')
            
            if not key_location:
                raise ValueError("PATH_TO_KEY environment variable is not set")
            
            key_path = os.path.join(key_location, 'encryption_key.key')
            
            if not os.path.exists(key_path):
                raise FileNotFoundError(f"Encryption key not found at {key_path}")

            with open(key_path, 'rb') as key_file:
                key = key_file.read()
            
            # Validate it's a proper Fernet key
            Fernet(key)
            
            return key
        
        except Exception as e:
            logging.error(f"Could not read encryption key: {e}")
            raise
        
    def encrypt_file(self, file_path, key):
        fernet = Fernet(key)
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
                
            encrypted_data = fernet.encrypt(file_data)
            
            with open(file_path + '.encrypted', 'wb') as file:
                file.write(encrypted_data)

        except Exception as e:
            logging.error(f"Encryption failed for {file_path}: {e}")
            return
        
        # Delete original unencrypted file
        os.remove(file_path)
    
    def decrypt_file(self, encrypted_file_path, key):
        
        fernet = Fernet(key)
        
        # Read encrypted data
        try:
            with open(encrypted_file_path, 'rb') as file:
                encrypted_data = file.read()
        except Exception as e:
            logging.error(f"Could not read from {encrypted_file_path}: {e}")
            return None
        
        # Decrypt data
        try:
            decrypted_data = fernet.decrypt(encrypted_data)
            decrypted_file_path = encrypted_file_path.replace('.encrypted', '')

        except Exception as e:
            logging.error(f"Decryption failed for {encrypted_file_path}: {e}")
            return None
        
        # Write decrypted data to new file
        try:
            with open(decrypted_file_path, 'wb') as file:
                file.write(decrypted_data)
        except Exception as e:
            logging.error(f"Could not write to {decrypted_file_path}: {e}")
            return None
        
        # Delete encrypted file
        os.remove(encrypted_file_path)
        
        return decrypted_file_path


class VideoManager(EncryptionManager):
    def __init__(self):
        load_dotenv()
        self.session = boto3.Session(
            region_name='auto')

        self.s3 = self.session.client(
            's3', 
            endpoint_url=os.getenv('R2_ENDPOINT'))

    def download_all(self, directory):
        # check if directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # get bucket name from env
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        continuation_token = None
        load_dotenv()

        encryption_key = self.get_encryption_key()
        fernet = Fernet(encryption_key)
        
        try:
            while True:
                list_kwargs = {'Bucket': self.bucket_name}
                if continuation_token:
                    list_kwargs['ContinuationToken'] = continuation_token
                    
                print("listing objects with args: ", list_kwargs)
                response = self.s3.list_objects_v2(**list_kwargs)
                
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    print("loading file: ", key)
                    if key.endswith('.mp4'):
                        try:
                            # Download to memory buffer
                            from io import BytesIO
                            buffer = BytesIO()
                            self.s3.download_fileobj(self.bucket_name, key, buffer)
                            
                            buffer.seek(0)
                            file_data = buffer.read()
                            
                            # Encrypt in memory
                            encrypted_data = fernet.encrypt(file_data)
                            
                            flat_filename = key.replace('/', '_') + '.encrypted'
                            
                            encrypted_path = os.path.join(directory, flat_filename)
                            with open(encrypted_path, 'wb') as f:
                                f.write(encrypted_data)
                            
                            logging.info(f'Downloaded and encrypted {key}')
                            
                        except Exception as e:
                            logging.error(f"Failed to download/encrypt {key}: {e}")
                            continue
                
                if not response.get('IsTruncated'):
                    break
                continuation_token = response.get('NextContinuationToken')     
        except Exception as e:
            logging.error(f"Error downloading files: {e}")
            raise

    def orginise_videos(self, unorginised_directory, orginised_directory):
        # Ensure the orginised directory exists
        if not os.path.exists(orginised_directory):
            os.makedirs(orginised_directory)
        
        for file_name in os.listdir(unorginised_directory):
            if file_name.endswith('.mp4.encrypted'):
                parts = file_name.split('_')

                # for format: submission/IDnumber/Letter.mp4.encrypted
                submission_id = parts[1]
                print(f'parts: {parts[2]}, submission_id: {submission_id}')
                if len(parts[2]) == 1:
                    # takes first char of [2]
                    letter = parts[2][0].upper()
                else:
                    # takes all of [2] except '.mp4.encrypted'
                    letter = parts[2][:-13].upper()
                
                # moves file in orginised_directory/letter
                target_directory = os.path.join(orginised_directory, letter)
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)
                
                source_path = os.path.join(unorginised_directory, file_name)
                target_path = os.path.join(target_directory, file_name)
                os.rename(source_path, target_path)
                print(f'Moved {source_path} to {target_path}')

    def delete_video(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"File {file_path} does not exist")
        return
            
    def loopthroughcorpus(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    # encrypt the file
                    file_path = os.path.join(root, file)
                    encryption_key = self.get_encryption_key()
                    self.encrypt_file(file_path, encryption_key)

    def quality_check(self, directory):
        # Collect all encrypted video files from letter folders
        video_files = []
        SKIP_DIRS = {"valid", "needs_attention", "damaged_inappropriate"}
        
        for root, dirs, files in os.walk(directory):
            # Skip category directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for file in files:
                if file.endswith('.mp4.encrypted'):
                    file_path = os.path.join(root, file)
                    video_files.append(file_path)
        
        if not video_files:
            print("No encrypted videos found for quality check")
            return
        
        print(f"Found {len(video_files)} videos to check")
        
        # Process each video
        current_index = [0]
        encryption_key = self.get_encryption_key()
        
        def process_next_video():
            # process the next video in the list
            if current_index[0] >= len(video_files):
                messagebox.showinfo("Complete", "All videos have been reviewed!")
                return
            
            encrypted_path = video_files[current_index[0]]
            
            # Get the letter from the parent folder name
            letter_folder = os.path.basename(os.path.dirname(encrypted_path))
            
            print(f"\n[{current_index[0] + 1}/{len(video_files)}] Processing: {encrypted_path}")
            print(f"  Letter: {letter_folder}")
            
            # Decrypt video
            try:
                decrypted_path = self.decrypt_file(encrypted_path, encryption_key)
                print(f"  Decrypted to: {decrypted_path}")
            except Exception as e:
                print(f"  Failed to decrypt: {e}")
                messagebox.showerror("Error", f"Failed to decrypt {os.path.basename(encrypted_path)}")
                current_index[0] += 1
                process_next_video()
                return
            
            # Show video player window
            self._show_video_player(
                decrypted_path, 
                letter_folder,  # Use folder name as letter
                encrypted_path,
                encryption_key,
                lambda: on_button_click('valid'),
                lambda: on_button_click('needs_attention'),
                lambda: on_button_click('damaged_inappropriate')
            )
        
        def on_button_click(category):
            """Handle button clicks for video categorization."""
            encrypted_path = video_files[current_index[0]]
            
            # Get the letter folder (parent directory)
            letter_folder = os.path.dirname(encrypted_path)
            
            # Create category folder inside the letter folder
            category_dir = os.path.join(letter_folder, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Move encrypted file to category folder
            filename = os.path.basename(encrypted_path)
            target_path = os.path.join(category_dir, filename)
            
            try:
                shutil.move(encrypted_path, target_path)
                print(f"  Moved to: {target_path}")
            except Exception as e:
                print(f"  Failed to move file: {e}")
                messagebox.showerror("Error", f"Failed to move file: {e}")
            
            # Move to next video
            current_index[0] += 1
            process_next_video()
        
        # Start processing
        process_next_video()

    def _show_video_player(self, video_path, letter, encrypted_path, encryption_key, 
                        on_valid, on_needs_attention, on_damaged):
        
        # Create window
        window = tk.Tk()
        window.title(f"Quality Check - Letter: {letter}")
        
        # Video display label
        video_label = tk.Label(window)
        video_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(window)
        button_frame.pack(pady=10)
        
        # Control variables
        is_playing = [True]
        cap = [None]
        
        def cleanup_and_close(callback):
            """Clean up resources and execute callback."""
            is_playing[0] = False
            if cap[0]:
                cap[0].release()
            
            # Re-encrypt the video
            try:
                print(f"  Re-encrypting...")
                self.encrypt_file(video_path, encryption_key)
                print(f"  Re-encrypted successfully")
            except Exception as e:
                print(f"  Failed to re-encrypt: {e}")
                messagebox.showerror("Error", f"Failed to re-encrypt: {e}")
            
            window.destroy()
            window.quit(0, callback())
        
        # Buttons
        valid_btn = tk.Button(
            button_frame, 
            text="Valid", 
            command=lambda: cleanup_and_close(on_valid),
            bg="green",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 12, "bold")
        )
        valid_btn.pack(side=tk.LEFT, padx=5)
        
        attention_btn = tk.Button(
            button_frame, 
            text="Needs Attention", 
            command=lambda: cleanup_and_close(on_needs_attention),
            bg="orange",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 12, "bold")
        )
        attention_btn.pack(side=tk.LEFT, padx=5)
        
        damaged_btn = tk.Button(
            button_frame, 
            text="Damaged/Inappropriate", 
            command=lambda: cleanup_and_close(on_damaged),
            bg="red",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 12, "bold")
        )
        damaged_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label showing letter and filename
        info_label = tk.Label(
            window, 
            text=f"Letter: {letter} | File: {os.path.basename(encrypted_path)}",
            font=("Arial", 10)
        )
        info_label.pack(pady=5)
        
        def play_video():
            """Play video in loop."""
            cap[0] = cv2.VideoCapture(video_path)
            
            if not cap[0].isOpened():
                messagebox.showerror("Error", "Failed to open video file")
                window.destroy()
                return
            
            def update_frame():
                if not is_playing[0]:
                    return
                
                ret, frame = cap[0].read()
                
                # Loop video
                if not ret:
                    cap[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap[0].read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize to fit window (max 800x600)
                    height, width = frame.shape[:2]
                    max_width = 800
                    max_height = 600
                    
                    if width > max_width or height > max_height:
                        scale = min(max_width/width, max_height/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to PhotoImage
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                
                # Schedule next frame (30 FPS)
                video_label.after(33, update_frame)
            
            update_frame()
        
        # Start video playback in separate thread
        threading.Thread(target=play_video, daemon=True).start()
        
        # Handle window close
        def on_closing():
            messagebox.showwarning(
                "Warning", 
                "Please use one of the buttons to categorize the video"
            )
        
        window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the GUI
        window.mainloop()
                    
    def submission_check(self, directory):
        # adds all submission encrypted videos to a list Ids found to list
        submission_ids = set()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4.encrypted'):
                    parts = file.split('_')
                    if len(parts) > 1:
                        submission_id = parts[1]
                        submission_ids.add(submission_id)
        
        # writes list to submission_ids.txt
        with open('submission_ids.txt', 'w') as f:
            for submission_id in sorted(submission_ids):
                f.write(f"{submission_id}\n")
        return list(submission_ids)
                    
                    
video_manager = VideoManager()
video_manager.loopthroughcorpus('videoCorpus_sorted')