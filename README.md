# IC_Detector
Repo for EECS 690 (783) IC detection project

Timeline - excle
https://kansas-my.sharepoint.com/:x:/g/personal/s803s199_home_ku_edu/ESS4Z3hTPQ9Gk2kHx_dROKIB3m23Sv2ybKYqHFhkZEuwjQ?e=Qo6MwE

draw.io
https://drive.google.com/file/d/1mfwxzjhkvvAMCvM5PCTR-OuLXJQopARx/view?usp=sharing

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Open the Service Accounts page

Open your browser and go to the Google Cloud Console. https://console.cloud.google.com/welcome/new

At the top, select the project you want to use (from the project dropdown in the upper-right corner).

Open the left-side menu (☰) → navigate to:
“IAM & Admin” → “Service Accounts”.

(Note: In some newer layouts, it may appear under “Menu → IAM & Admin → Service Accounts.”)

2. Create a JSON key for this service account (the file you need)

In the Service Accounts list, click the email address of the service account you want to use.

In the account details page, go to the “Keys” tab.

Click “Add key” → “Create new key.”

A dialog box will pop up asking for Key type.
Choose JSON, then click Create.

Your browser will automatically download a .json file, for example:
glowing-river-123456-0a1b2c3d4e5f.json

This file is your service account key.json.

(As stated in the official docs: Service account details → Keys → Add key → Create new key → Select JSON → Create → Browser downloads JSON file.)

3. Place the JSON file in an easy-to-find location

It’s recommended to create a simple directory path, such as:

D:\gcp_keys\vision-key.json

Move the downloaded .json file there and remember the full path.

4.Tell Python / the SDK to use this file

In the same command prompt (CMD or Google Cloud SDK Shell) where you’ll run your Python script (python main.py), set the environment variable:

setx GOOGLE_APPLICATION_CREDENTIALS "D:\gcp_keys\vision-key.json"


Finally, enable the Vision API at:
https://console.developers.google.com/apis/api/vision.googleapis.com/overview