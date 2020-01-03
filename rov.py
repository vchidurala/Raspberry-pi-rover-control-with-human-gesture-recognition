import time
from flask import Flask, request
from AMSpi import AMSpi
app  = Flask(__name__)

def xx(x):
###########################AMSpi.pyc is a python class file for l293d/ shift register module that we are using in rover #####
	with AMSpi() as amspi:
#amspi = AMSpi()
		print("helloo..")
		amspi.set_74HC595_pins(21, 20, 16)

		amspi.set_L293D_pins(5, 6, 13, 19)
		if(x=='forward'):
			amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_2, amspi.DC_Motor_3, amspi.DC_Motor_4])
			time.sleep(2)
		elif(x=='left'):
			amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_3],clockwise=False)
			amspi.run_dc_motors([amspi.DC_Motor_2, amspi.DC_Motor_4])
			time.sleep(10)
		elif(x=='right'):
			amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_3])
			amspi.run_dc_motors([amspi.DC_Motor_2, amspi.DC_Motor_4],clockwise=False)
			time.sleep(10)

		elif(x=='back'):
			amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_2, amspi.DC_Motor_3, amspi.DC_Motor_4],clockwise=False)
			time.sleep(2)
		elif(x=='stop'):
			amspi.stop_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_2, amspi.DC_Motor_3, amspi.DC_Motor_4])

		elif(x=='Square'):
                        for i in range(1,4):
				print("going straight...")
				amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_2, amspi.DC_Motor_3, amspi.DC_Motor_4])
        	                time.sleep(2)
				print("turning left...")
				amspi.run_dc_motors([amspi.DC_Motor_1, amspi.DC_Motor_3],clockwise=False)
                        	amspi.run_dc_motors([amspi.DC_Motor_2, amspi.DC_Motor_4])
	                        time.sleep(10)



		print("yoyobaba..rockzz..")

#xx()
@app.route("/", methods=['POST'])
def result():
	print(request.form['direction'])
	xx(request.form['direction'])
	return('Recieved..!' + request.form['direction'])
if __name__=="__main__":
	app.run(host='0.0.0.0', port=80, debug=True)
