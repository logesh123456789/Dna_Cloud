# main.py
import os
import base64
#from cryptography.hazmat.backends import default_backend
#from cryptography.hazmat.primitives import hashes
#from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
#from cryptography.fernet import Fernet
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import shutil
import datetime
import random
from random import randint
import matplotlib.pyplot as plt
from flask_mail import Mail, Message
from flask import send_file
from werkzeug.utils import secure_filename
from PIL import Image
import stepic
import urllib.parse
from urllib.request import urlopen
import webbrowser
import socket

#from ldpc.encoder.base_encoder import Encoder
#from numpy.typing import NDArray
#import numpy as np
#from ldpc.utils.custom_exceptions import IncorrectLength, NonBinaryMatrix


#img
#import pywhatkit as kt
##wrd to htm
import mammoth

#pdf
from pdf2docx import parse
from typing import Tuple
import pdfcrowd
import sys

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="dna_cloud"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM dc_register WHERE uname = %s AND pass = %s && status=1', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('upload'))
        else:
            msg = 'Incorrect username/password! or not approved!'
    return render_template('index.html',msg=msg)

@app.route('/login_csp', methods=['GET', 'POST'])
def login_csp():
    msg=""

    
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM dc_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_csp.html',msg=msg)



@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM dc_user WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM dc_register")
    maxid = mycursor.fetchone()[0]

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
            
    if maxid is None:
        maxid=1
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        city=request.form['city']
        uname=request.form['uname']
        pass1=request.form['pass']
        cursor = mydb.cursor()

        cursor.execute('SELECT count(*) FROM dc_register WHERE uname = %s ', (uname,))
        cnt = cursor.fetchone()[0]
        if cnt==0:
            result = hashlib.md5(uname.encode())
            key=result.hexdigest()
            pbkey=key[0:8]
            prkey=key[8:16]
            sql = "INSERT INTO dc_register(id,name,mobile,email,city,public_key,private_key,uname,pass,rdate,status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,name,mobile,email,city,pbkey,prkey,uname,pass1,rdate,'0')
            cursor.execute(sql, val)
            mydb.commit()            
            print(cursor.rowcount, "Registered Success")
            msg="success"

            ##send mail
            #mess="Owner:"+uname+", Public Key:"+pbkey+", Private Key:"+prkey
            
            #if cursor.rowcount==1:
            #return redirect(url_for('index'))
        else:
            msg='fail'
    return render_template('register.html',msg=msg)



############s
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    email=""
    mess=""
    act=request.args.get("act")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register')
    data=cursor1.fetchall()

    if act=="ok":
        uid=request.args.get("uid")
        cursor1.execute('SELECT * FROM dc_register where id=%s',(uid,))
        dd=cursor1.fetchone()
        owner=dd[7]
        pbkey=dd[5]
        prkey=dd[6]
        email=dd[3]
        mess="Owner:"+owner+", Public Key:"+pbkey+", Private Key:"+prkey
        print(mess)
        cursor1.execute("update dc_register set status=1 where id=%s",(uid,))
        mydb.commit()
        msg="ok"

    return render_template('admin.html',act=act,msg=msg,data=data,email=email,mess=mess)

def convert_pdf2docx(input_file: str, output_file: str, pages: Tuple = None):
    """Converts pdf to docx"""
    if pages:
        pages = [int(i) for i in list(pages) if i.isnumeric()]
    result = parse(pdf_file=input_file,
                   docx_with_path=output_file, pages=pages)
    summary = {
        "File": input_file, "Pages": str(pages), "Output File": output_file
    }
    # Printing Summary
    print("## Summary ########################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in summary.items()))
    print("###################################################################")
    return result
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    act=""
    fid=""
    fsize=0
    uname=""
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register where uname=%s',(uname, ))
    rr=cursor1.fetchone()
    name=rr[1]
    pbkey = rr[5]
    email = rr[3]
    #pbkey=data1[9]
    
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    if request.method=='POST':
        file_content=request.form['content']
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM dc_user_files")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        fid=str(maxid)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = "F"+str(maxid)+file.filename
            filename = secure_filename(fname)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fsize1=os.path.getsize("static/upload/"+filename)
            fsize=fsize1/1024

            
        ##encryption
        '''password_provided = pbkey # This is input in the form of a string
        password = password_provided.encode() # Convert to type bytes
        salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))

        input_file = 'static/upload/'+fname
        output_file = 'static/upload/E'+fname
        with open(input_file, 'rb') as f:
            data = f.read()

        fernet = Fernet(key)
        encrypted = fernet.encrypt(data)

        with open(output_file, 'wb') as f:
            f.write(encrypted)'''
            
        
        
        ##store
        sql = "INSERT INTO dc_user_files(id,uname,file_type,file_content,upload_file,rdate,filesize1) VALUES (%s, %s, %s, %s, %s, %s,%s)"
        val = (maxid,uname,file_type,file_content,filename,rdate,fsize)
        mycursor.execute(sql,val)
        mydb.commit()

        ######
        filename2="H"+str(maxid)+".html"
        fs=filename.split('.')

        if fs[1]=="docx":
            s=1
            custom_styles = "b => i"
            input_filename="static/upload/"+filename
            with open(input_filename, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file, style_map = custom_styles)
                text = result.value
                with open("static/upload/"+filename2, 'w') as html_file:
                    html_file.write(text)

            ##ascii
            ff=open("static/upload/"+filename2,"r")
            fdata=ff.read()
            ff.close()

            
            
            fn="A"+fs[0]+".txt"
            dval=''.join(str(ord(c)) for c in fdata)

            ff=open("static/upload/"+fn,"w")
            ff.write(dval)
            ff.close()

            lf=len(dval)
            lf2=int(lf/4)
            fdata1=dval[0:lf2]
            

            ##binary
            line2=''.join(format(ord(x), 'b') for x in fdata1)
            bstr2=line2.encode('utf-8')
            dval2=bstr2.decode()

            fn2="B"+fs[0]+".txt"
            ff=open("static/upload/"+fn2,"w")
            ff.write(dval2)
            ff.close()
            #####

                    
        elif fs[1]=="pdf":
            s=1
            print("pdf")
            filename4="D"+str(maxid)+".docx"
            input_file = "static/upload/"+filename
            output_file = "static/upload/"+filename4
            convert_pdf2docx(input_file, output_file)

            fg=filename.split('.')
            fg2=fg[0]+".docx"

            custom_styles = "b => i"
            input_filename="static/upload/"+fg2
            with open(input_filename, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file, style_map = custom_styles)
                text = result.value
                with open("static/upload/"+filename2, 'w') as html_file:
                    html_file.write(text)

            ##ascii
            ff=open("static/upload/"+filename2,"r")
            fdata=ff.read()
            ff.close()

            
            fn="A"+fs[0]+".txt"
            dval=''.join(str(ord(c)) for c in fdata)

            ff=open("static/upload/"+fn,"w")
            ff.write(dval)
            ff.close()
            
            lf=len(dval)
            lf2=int(lf/4)
            fdata1=dval[0:lf2]
            ##binary
            line2=''.join(format(ord(x), 'b') for x in fdata1)
            bstr2=line2.encode('utf-8')
            dval2=bstr2.decode()

            fn2="B"+fs[0]+".txt"
            ff=open("static/upload/"+fn2,"w")
            ff.write(dval2)
            ff.close()
            #####

                    
        elif fs[1]=="jpg" or fs[1]=="png":
            s=1
            import pywhatkit as kt
            source_path = filename
            
            target_path = "F"+fs[0]+".txt"
            kt.image_to_ascii_art("static/upload/"+source_path, "static/upload/"+target_path)

            ##            
            ff=open("static/upload/"+target_path,"r")
            fdata=ff.read()
            ff.close()
            ##
            fn="A"+fs[0]+".txt"
            dval=''.join(str(ord(c)) for c in fdata)

            ff=open("static/upload/"+fn,"w")
            ff.write(dval)
            ff.close()

            lf=len(dval)
            lf2=int(lf/4)
            fdata1=dval[0:lf2]
            ##binary
            line2=''.join(format(ord(x), 'b') for x in fdata1)
            bstr2=line2.encode('utf-8')
            dval2=bstr2.decode()

            fn2="B"+fs[0]+".txt"
            ff=open("static/upload/"+fn2,"w")
            ff.write(dval2)
            ff.close()
            #####

        else:
            ##ascii
            ff=open("static/upload/"+filename,"r")
            fdata=ff.read()
            ff.close()

            
            fn="A"+fs[0]+".txt"
            dval=''.join(str(ord(c)) for c in fdata)

            ff=open("static/upload/"+fn,"w")
            ff.write(dval)
            ff.close()
            

            ##binary
            line2=''.join(format(ord(x), 'b') for x in fdata)
            bstr2=line2.encode('utf-8')
            dval2=bstr2.decode()

            fn2="B"+fs[0]+".txt"
            ff=open("static/upload/"+fn2,"w")
            ff.write(dval2)
            ff.close()
            #####

        
        
        msg="ok"
        #return redirect(url_for('upload_st',fname=filename))
            
        
    
    return render_template('upload.html',msg=msg,rr=rr,name=name,fid=fid)


@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""
    fid=request.args.get("fid")
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="A"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()

    i=0
    s1=""
    for s in fdata:
        if i<30:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1
    
    return render_template('process1.html',msg=msg,fdata=fdata,s1=s1,fid=fid)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    
    msg=""
    fid=request.args.get("fid")
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()


    i=0
    s1=""
    for s in fdata:
        if i<8:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1
    
    return render_template('process2.html',msg=msg,fdata=fdata,s1=s1,fid=fid)

@app.route('/process3', methods=['GET', 'POST'])
def process3():
    msg=""
    fdata=""
    s1=""
    fid=request.args.get("fid")
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()

    ##DNA
    hashval=fdata
    #print(hashval)
    #Substitutional Algorithm
    lv=len(hashval)
    gg=""
    i=0
    while i<lv-2:
        g=hashval[i]
        h=hashval[i+1]
        gh=g+""+h
        print(gh)
        if gh=="00":
            gg+="A"
        elif gh=="01":
            gg+="C"
        elif gh=="10":
            gg+="G"
        elif gh=="11":
            gg+="T"
        i+=2
    dval3=gg

    i=0
    s1=""
    for s in dval3:
        if i<10:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1
    
    return render_template('process3.html',msg=msg,fdata=fdata,s1=s1,fid=fid)

@app.route('/process4', methods=['GET', 'POST'])
def process4():
    msg=""
    fdata=""
    s1=""
    st=""
    fid=request.args.get("fid")
    
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register where uname=%s',(uname, ))
    rr=cursor1.fetchone()
    name=rr[1]
    pbkey = rr[5]
    email = rr[3]
    
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()

    if request.method=='POST':
        
        pbk=request.form['pbk']

        if pbk==pbkey:
            st="1"
            ##DNA
            hashval=fdata
            #print(hashval)
            #Substitutional Algorithm
            lv=len(hashval)
            gg=""
            i=0
            while i<lv-2:
                g=hashval[i]
                h=hashval[i+1]
                gh=g+""+h
                print(gh)
                if gh=="00":
                    gg+="A"
                elif gh=="01":
                    gg+="C"
                elif gh=="10":
                    gg+="G"
                elif gh=="11":
                    gg+="T"
                i+=2
            dval3=gg
            ##Encrypt
            x1=len(dval3)
            s1="CAGTAGA"
            s2="GCCATCG"
            s3="GTTCAAT"
            s4="CTCAGTG"
            j=0
            kk=""
            while j<x1:
                if dval3[j]=="A":
                    kk+="A"+s1
                elif dval3[j]=="C":
                    kk+="C"+s2
                elif dval3[j]=="G":
                    kk+="G"+s3
                elif dval3[j]=="T":
                    kk+="T"+s4
                j+=1
            dval4=kk
            
            i=0
            s1=""
            for s in dval4:
                if i<8:
                    s1+=s
                else:
                    s1+=" "
                    i=0

                i+=1

        else:
            msg="Public Key Incorrect!"
    
    return render_template('process4.html',msg=msg,fdata=fdata,s1=s1,st=st,fid=fid)

#LDBC Encoder
def Ldbc():
   
    self.h = h
    if h.max(initial=0) > 1 or h.min(initial=1) < 0:
        raise NonBinaryMatrix
    m, n = h.shape
    k = n - m
    self.m = m
    # check if parity part is identity
    self.identity_p = np.array_equal(h[:, k:], np.identity(m))

    super().__init__(k, n)

def encode():
    if len(information_bits) != self.k:
        raise IncorrectLength
    encoded: NDArray[np.int_] = np.zeros(self.n, dtype=np.int_)
    encoded[:self.k] = information_bits
    p: NDArray[np.int_] = np.mod(np.matmul(self.h[:, :self.k], information_bits), 2)
    if not self.identity_p:
        for l in range(1, self.m):
            p[l] += np.mod(np.dot(self.h[l, self.k:self.k+l], p[:l]), 2)
    encoded[self.k:] = p
    return encoded

def EncoderG(Encoder):
    
    self.generator = generator
    if generator.max(initial=0) > 1 or generator.min(initial=1) < 0:
        raise NonBinaryMatrix
    k, n = generator.shape
    super().__init__(k, n)

def encodeGenerator():
    if len(information_bits) != self.k:
        raise IncorrectLength
    return np.matmul(np.array(information_bits, dtype=np.int_), self.generator) % 2  # type: ignore

def Encoder(Encoder):
    
    self.spec = spec
    qc_file = QCFile.from_file(os.path.join(self._spec_base_path, spec.name + ".qc"))
    self.h = qc_file.to_array()
    self.m, n = self.h.shape
    k = n - self.m
    self.z = qc_file.z
    self.block_structure = qc_file.block_structure
    super().__init__(k, n)

def ldbc_encode():
    
    if len(information_bits) != self.k:
        raise IncorrectLength

    shifted_messages = self._shifted_messages(information_bits)
    parities: NDArray[np.int_] = np.zeros((self.m//self.z, self.z), dtype=np.int_)
    # special parts see article
    parities[0, :] = np.sum(shifted_messages, axis=0) % 2  # find first batch of z parity bits
    parities[1, :] = (shifted_messages[0, :] + np.roll(parities[0, :], -1)) % 2  # find second set of z parity bits
    parities[-1, :] = (shifted_messages[-1, :] + np.roll(parities[0, :], -1)) % 2  # find last set of z parity bits
    for idx in range(1, (self.m//self.z)-2):  # -1 needed to avoid exceeding memory limits due to idx+1 below.
        # -2 needed as bottom row is a special case.
        if self.block_structure[idx][self.k // self.z] >= 0:
            # special treatment of x-th row, see article
            parities[idx+1, :] = (parities[idx, :] + shifted_messages[idx, :] + parities[0, :]) % 2
        else:
            parities[idx+1, :] = (parities[idx, :] + shifted_messages[idx, :]) % 2

    return np.concatenate((information_bits, np.ravel(parities)))

def _shifted_messages():
    # break message bits into groups (rows) of Z bits. Each row is a subset of z bits, overall k message bits
    bit_blocks: NDArray[np.int_] = information_bits.reshape((self.k // self.z, self.z))

    # find shifted messages (termed lambda_i in article)
    shifted_messages: NDArray[np.int_] = np.zeros((self.m // self.z, self.z),
                                                  dtype=np.int_)  # each row is a sum of circular shifts of
    # message bits (some lambda_i in article). One row per block of h.
    for i in range(self.m // self.z):
        for j in range(self.k // self.z):
            if self.block_structure[i][j] >= 0:  # zero blocks don't contribute to parity bits
                # multiply by translation reduces to shift.
                vec: NDArray[Any] = np.roll(bit_blocks[j, :], -self.block_structure[i][j])
                shifted_messages[i, :] = np.logical_xor(shifted_messages[i, :], vec)  # xor as sum mod 2
    return shifted_messages

############
#LDBC Decoder
def LdbcDecoder():
    
    self.info_idx = info_idx
    self.h: NDArray[np.int_] = h
    self.m, self.n = h.shape
    self.k = self.n - self.m if info_idx is None else np.sum(info_idx)
    self.max_iter = max_iter
    self.percent_flipped = percent_flipped

def decode():
    if len(channel_word) != self.n:
        raise IncorrectLength("incorrect block size")
    if min(channel_word) < 0:  # LLR values were given
        channel_word = np.array(channel_word < 0, dtype=np.int_)
    else:
        channel_word = channel_word.astype(np.int_)
    vnode_validity = np.zeros(self.n, dtype=np.int_)
    for iteration in range(self.max_iter):
        syndrome = self.h @ channel_word % 2
        if not syndrome.any():  # no errors detected, exit
            break
        # for each vnode how many equations are failed
        vnode_validity = syndrome @ self.h
        num_suspected_vnodes = sum(vnode_validity > 0)
        num_flip_bits = 1  # max(1, num_suspected_vnodes//self.percent_flipped)  # flip 10% of the suspected bits
        flip_bits = np.argpartition(vnode_validity, -num_flip_bits)[-num_flip_bits:]
        channel_word[flip_bits] = 1 - channel_word[flip_bits]

    return channel_word, not syndrome.any(), iteration+1, syndrome, vnode_validity

def info_bits():
    """extract information bearing bits from decoded estimate, assuming info bits indices were specified"""
    if self.info_idx is not None:
        return estimate[self.info_idx]
    else:
        raise InfoBitsNotSpecified("decoder cannot tell info bits")

def LogSpaDecoder():
   
    self.decoder_type = decoder_type
    self.info_idx = info_idx
    self.h: npt.NDArray[np.int_] = h
    self.graph = TannerGraph.from_biadjacency_matrix(h=self.h, channel_model=channel_model, decoder=decoder_type)
    self.n = len(self.graph.v_nodes)
    self.max_iter = max_iter
    ordered_cnodes = sorted(self.graph.c_nodes.values())
    self.ordered_cnodes_uids = [node.uid for node in ordered_cnodes]
    self._ordered_vnodes = self.graph.ordered_v_nodes()

def ldbc_decode():
    
        self.decoder_type = decoder_type
        self.info_idx = info_idx
        self.h: npt.NDArray[np.int_] = h
        self.graph = TannerGraph.from_biadjacency_matrix(h=self.h, channel_model=channel_model, decoder=decoder_type)
        self.n = len(self.graph.v_nodes)
        self.max_iter = max_iter
        ordered_cnodes = sorted(self.graph.c_nodes.values())
        self.ordered_cnodes_uids = [node.uid for node in ordered_cnodes]
        self._ordered_vnodes = self.graph.ordered_v_nodes()

    
        if len(channel_word) != self.n:
            raise IncorrectLength("incorrect block size")

        # initial step
        for idx, vnode in enumerate(self._ordered_vnodes):
            vnode.initialize(channel_word[idx])
        for cnode in self.graph.c_nodes.values():  # send initial channel based messages to check nodes
            cnode.receive_messages()

        if max_iter is None:
            max_iter = self.max_iter
        for iteration in range(max_iter):
            # Check to Variable Node Step(horizontal step):
            for vnode in self.graph.v_nodes.values():
                vnode.receive_messages()
            # Variable to Check Node Step(vertical step)
            for cnode in self.graph.c_nodes.values():
                cnode.receive_messages()

            # Check stop condition
            llr: npt.NDArray[np.float_] = np.array([node.estimate() for node in self._ordered_vnodes], dtype=np.float_)
            estimate: npt.NDArray[np.int_] = np.array(llr < 0, dtype=np.int_)
            syndrome = self.h @ estimate % 2
            if not syndrome.any():
                break

        # for each vnode how many equations are failed
        vnode_validity: npt.NDArray[np.int_] = syndrome @ self.h  # type: ignore
        return estimate, llr, not syndrome.any(), iteration+1, syndrome, vnode_validity

def info_bits():
    if self.info_idx is not None:
        return estimate[self.info_idx]
    else:
        raise InfoBitsNotSpecified("decoder cannot tell info bits")

def ordered_vnodes():
    """getter for ordered graph v-nodes"""
    return self._ordered_vnodes

def update_channel_model():
    for uid, model in channel_models.items():
        node = self.graph.v_nodes.get(uid)
        if isinstance(node, VNode):
            node.channel_model = model

##########
    
@app.route('/process5', methods=['GET', 'POST'])
def process5():
    msg=""
    fdata=""
    s1=""
    fid=request.args.get("fid")
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="A"+fs[0]+".txt"

    ff=open("static/dna.txt","w")
    ff.write("")
    ff.close()

    ff2=open("static/dna2.txt","w")
    ff2.write("")
    ff2.close()
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()

    lf=len(fdata)
    lf2=int(lf/6)
    fdata1=fdata[0:lf2]

            
    mlen=len(fdata1)
    
    print("mlen")
    print(mlen)
    if mlen>100:
        mval=mlen/40
        mvalue=int(mval)
    else:
        mval=mlen/10
        mvalue=int(mval)

        
    return render_template('process5.html',msg=msg,fdata=fdata,s1=s1,fid=fid,mvalue=mvalue)

@app.route('/dna_visual', methods=['GET', 'POST'])
def dna_visual():
    act1=""
    fid=request.args.get("fid")
    st=""
    dd=[]
    
    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor1.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="C"+fs[0]+".fasta"
    dff="D"+fs[0]+".txt"

    
    #DNA Strain Generation
    mvalue=request.args.get("mvalue")
    act=request.args.get("act")
    if act is None:
        act="1"
    print("mvalue")
    print(mvalue)
    m=int(mvalue)
    n=int(act)
    data=[]
    PAUSE = 0.15  # Change it 0.0 and see what happen

    #ss=[1,2,3,4]
    ss=[]
    rw=[9,8,7,6,5,4,4,5,6,7,6,6,5,5,6,7,8,9]
    rlen=len(rw)-1

    
    ROWS = [
        '         ##',
        '        #{}-{}#',
        '       #{}---{}#',
        '      #{}-----{}#',
        '     #{}------{}#',
        '    #{}------{}#',
        '    #{}-----{}#',
        '     #{}---{}#',
        '      #{}-{}#',
        '       ##',
        '      #{}-{}#',
        '      #{}---{}#',
        '     #{}-----{}#',
        '     #{}------{}#',
        '      #{}------{}#',
        '       #{}-----{}#',
        '        #{}---{}#',
        '         #{}-{}#',]

    try:
        
        if n<=m:
            st="1"
            print('DNA Visualization || Ihtesham Haider')
            #print('Press CTRL-C on Keyboard to quit....')
            #time.sleep(2)
            rowIndex = 0
            
            s=1
            n+=3
            act1=str(n)
            i=0
            k=0
            spc=[]
            #Main loop of the program || Started
            while i<n:

                dt=[]
                if k<rlen:
                    k+=1
                else:
                    
                    k=0
                nr=rw[k]
                j=1
                spc=[]
                while j<=nr:
                    spc.append('1')
                    
                    j+=1
                
                #incrementing for to draw a next row:
                rowIndex = rowIndex +1
                if rowIndex == len(ROWS):
                    rowIndex = 0

                # Row indexes 0 and 9 don't have nucleotides:
                if rowIndex == 0 or rowIndex ==9:
                    print(ROWS[rowIndex])
                    continue



                randomSelection = random.randint(1,4)
                if randomSelection ==1:
                    leftNucleotide, rightNucleotide = 'A', 'T'
                elif randomSelection ==2:
                    leftNucleotide, rightNucleotide = 'T', 'A'
                elif randomSelection ==3:
                    leftNucleotide, rightNucleotide = 'C', 'G'
                elif randomSelection ==4:
                    leftNucleotide, rightNucleotide = 'G', 'C'

                # priting the row
                #print(ROWS[rowIndex].format(leftNucleotide, rightNucleotide))
                dd=ROWS[rowIndex].format(leftNucleotide, rightNucleotide)

                
                ff=open("static/dna.txt","a")
                
                ff.write("\n"+dd)
                ff.close()


                ff2=open("static/dna2.txt","a")
                ff2.write(dd+"|")
                ff2.close()

                
                shutil.copy("static/dna2.txt","static/upload/"+dff)
                
                dt.append(spc)
                dt.append(dd)
                data.append(dt)
                
                i+=1
            
            #fasta
            #shutil.copy("static/dna.txt","static/dna.fa")
            
        else:
            ff=open("static/dna2.txt","r")
            ddd=ff.read()
            ff.close()
            arr1=ddd.split("|")
            
            alen=len(arr1)
            ########################
            #File input
            fileInput = open("static/dna.txt", "r")

            #File output
            fn="dna.fasta"
            fileOutput = open("static/upload/"+fname, "w")

            #Seq count
            count = 1 

            #Loop through each line in the input file
            print("Converting to FASTA...")
            for strLine in fileInput:

                #Strip the endline character from each input line
                strLine = strLine.rstrip("\n")

                #Output the header
                fileOutput.write(">" + str(count) + "\n")
                fileOutput.write(strLine + "\n")

                count = count + 1
            print ("Done.")

            #Close the input and output file
            fileInput.close()
            fileOutput.close()

            fsize1=os.path.getsize("static/upload/"+fname)
            fsize=fsize1/1024
            cursor1.execute('update dc_user_files set fastafile=%s,filesize2=%s where id=%s',(fname,fsize,fid))
            mydb.commit()
            #######################
            k=0
            i=0
            dat=[]
            x=0
            '''with open('static/dna.txt') as f:
                dline=f.readlines()
                gs=str(dline)
                dline1=gs.rstrip('\n')
                dat.append(gs)
                x+=1'''
               

            for dat1 in arr1:
                if i<alen:
                    
                    print("k=")
                    print(k)
                    nr=rw[k]

                    if k<rlen-1:
                        k+=1
                    else:
                        k=0
                    j=1
                    spc1=[]
                    while j<=nr:
                        spc1.append('1')
                        
                        j+=1

                        
                    df=[]
                    df.append(spc1)
                    
                    dg = dat1.strip()
                    
                    df.append(dg)
                    dd.append(df)

                
                
                i+=1
                
    
            st="2"
    except KeyboardInterrupt:
        sys.exit()

        
        

    return render_template('dna_visual.html',ss=ss,data=data,act=act1,st=st,dd=dd,mvalue=mvalue,fn=fn,fid=fid)

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    fid=request.args.get("fid")
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()

    f1=float(rr[6])
    f11=f1/1024
    f12=f11/1024
    v1=f12
    v2=10-f12

    h1=float(rr[8])
    h11=h1/1024
    h12=h11/1024
    v3=h12
    v4=10-h12
    
    ##########
    #pie--appliances
    bval=['Uploaded File','Available Space']
     
    gdata = [v1,v2]
     
    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(gdata, labels = bval)
     
    # show plot
    #plt.show()
    fn="g"+fid+".png"
    plt.savefig('static/graph/'+fn)
    plt.close()
    ##########
    #pie--appliances
    bval2=['Fasta File','Available Space']
     
    gdata2 = [v3,v4]
     
    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(gdata2, labels = bval2)
     
    # show plot
    #plt.show()
    fn2="gf"+fid+".png"
    plt.savefig('static/graph/'+fn2)
    plt.close()

    return render_template('graph.html',fid=fid)

@app.route('/view_graph', methods=['GET', 'POST'])
def view_graph():
    act1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()

    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    data=cursor.fetchone()

    x11=float(data[6])
    x1=x11/1024
    x22=float(data[8])
    x2=x22/1024

    
    gn="g"+fid+".png"
    gn2="gf"+fid+".png"
    

    return render_template('view_graph.html',fid=fid,data=data,gn=gn,gn2=gn2,x1=x1,x2=x2)

@app.route('/dna_show', methods=['GET', 'POST'])
def dna_show():
    act1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    st=""
    dd=[]
    
    rw=[9,8,7,6,5,4,4,5,6,7,6,6,5,5,6,7,8,9]
    rlen=len(rw)-1
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="D"+fs[0]+".txt"
    #######################

    ff=open("static/upload/"+fname,"r")
    ddd=ff.read()
    ff.close()
    arr1=ddd.split("|")
    
    alen=len(arr1)
    k=0
    i=0
    dat=[]
    x=0
    '''with open('static/dna.txt') as f:
        dline=f.readlines()
        gs=str(dline)
        dline1=gs.rstrip('\n')
        dat.append(gs)
        x+=1'''
       

    for dat1 in arr1:
        if i<alen:
            
            print("k=")
            print(k)
            nr=rw[k]

            if k<rlen-1:
                k+=1
            else:
                k=0
            j=1
            spc1=[]
            while j<=nr:
                spc1.append('1')
                
                j+=1

                
            df=[]
            df.append(spc1)
            
            dg = dat1.strip()
            
            df.append(dg)
            dd.append(df)

        
        
        i+=1
        

    st="2"
    

    return render_template('dna_show.html',st=st,dd=dd,fn=fn,fid=fid)

    
@app.route('/user_view', methods=['GET', 'POST'])
def user_view():
    act1=""
    msg=""
    mess=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="C"+fs[0]+".fasta"

    if request.method=='POST':

        rn=randint(1000,9999)
        kk=str(rn)
        result = hashlib.md5(kk.encode())
        key=result.hexdigest()
        skey=key[0:8]
        mess="Secret Key: "+skey

        cursor.execute("update dc_user set secret_key=%s where uname=%s",(skey,uname))
        mydb.commit()

        msg="ok"
    
        
        

    return render_template('user_view.html',msg=msg,act=act1,fid=fid,mess=mess,email=email)

@app.route('/decode1', methods=['GET', 'POST'])
def decode1():
    act1=""
    msg=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="C"+fs[0]+".fasta"

    if request.method=='POST':
        skey=request.form['skey']
        if key==skey:

            msg="ok"
        else:
            msg="fail"
        
        

    return render_template('decode1.html',msg=msg,act=act1,fid=fid)


@app.route('/decode2', methods=['GET', 'POST'])
def decode2():
    act1=""
    msg=""
    st=""
    s1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()
    hashval=fdata
    #print(hashval)
    #Substitutional Algorithm
    lv=len(hashval)
    gg=""
    i=0
    while i<lv-2:
        g=hashval[i]
        h=hashval[i+1]
        gh=g+""+h
        print(gh)
        if gh=="00":
            gg+="A"
        elif gh=="01":
            gg+="C"
        elif gh=="10":
            gg+="G"
        elif gh=="11":
            gg+="T"
        i+=2
    dval3=gg
    ##Decrypt
    x1=len(dval3)
    s1="CAGTAGA"
    s2="GCCATCG"
    s3="GTTCAAT"
    s4="CTCAGTG"
    j=0
    kk=""
    while j<x1:
        if dval3[j]=="A":
            kk+="A"+s1
        elif dval3[j]=="C":
            kk+="C"+s2
        elif dval3[j]=="G":
            kk+="G"+s3
        elif dval3[j]=="T":
            kk+="T"+s4
        j+=1
    dval4=kk
    
    i=0
    s1=""
    st="1"
    for s in dval4:
        if i<8:
            
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1

        

    return render_template('decode2.html',msg=msg,act=act1,fid=fid,s1=s1,st=st)

@app.route('/decode3', methods=['GET', 'POST'])
def decode3():
    act1=""
    msg=""
    st=""
    s1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()
    
    ##DNA
    hashval=fdata
    #print(hashval)
    #Substitutional Algorithm
    lv=len(hashval)
    gg=""
    i=0
    while i<lv-2:
        g=hashval[i]
        h=hashval[i+1]
        gh=g+""+h
        print(gh)
        if gh=="00":
            gg+="A"
        elif gh=="01":
            gg+="C"
        elif gh=="10":
            gg+="G"
        elif gh=="11":
            gg+="T"
        i+=2
    dval3=gg

    i=0
    st="1"
    s1=""
    for s in dval3:
        if i<10:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1

    return render_template('decode3.html',msg=msg,act=act1,fid=fid,s1=s1,st=st)

@app.route('/decode4', methods=['GET', 'POST'])
def decode4():
    act1=""
    msg=""
    st=""
    s1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()
    
    i=0
    st="1"
    s1=""
    for s in fdata:
        if i<8:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1

    return render_template('decode4.html',msg=msg,act=act1,fid=fid,s1=s1,st=st)

@app.route('/decode5', methods=['GET', 'POST'])
def decode5():
    act1=""
    msg=""
    st=""
    s1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="A"+fs[0]+".txt"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()

    i=0
    s1=""
    st="1"
    for s in fdata:
        if i<30:
            s1+=s
        else:
            s1+=" "
            i=0

        i+=1

    return render_template('decode5.html',msg=msg,act=act1,fid=fid,s1=s1,st=st)

@app.route('/decode6', methods=['GET', 'POST'])
def decode6():
    act1=""
    msg=""
    st=""
    s1=""
    fid=request.args.get("fid")
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    email=rr[6]
    key=rr[11]
    st=""
    dd=[]
    
    
    cursor.execute('SELECT * FROM dc_user_files where id=%s',(fid, ))
    rr=cursor.fetchone()
    fn=rr[4]
    fs=fn.split('.')
    fname="B"+fs[0]+".txt"
    st="1"
    
    ff=open("static/upload/"+fname,"r")
    fdata=ff.read()
    ff.close()
    


    return render_template('decode6.html',msg=msg,act=act1,fid=fid,s1=s1,st=st,fname=fn)

@app.route('/upload_st', methods=['GET', 'POST'])
def upload_st():
    msg=""
    act=""
    if 'username' in session:
        uname = session['username']
    print(uname)

    fname = request.args.get('fname')
    
    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register where uname=%s',(uname, ))
    rr=cursor1.fetchone()
    name=rr[1]
    pbkey = rr[5]
    email = rr[3]
    #pbkey=data1[9]

    data1="Owner: "+uname+", File Upload, File: "+fname


    return render_template('upload_st.html',msg=msg,name=name,data1=data1)


@app.route('/view_files', methods=['GET', 'POST'])
def view_files():
    msg=""
    act = request.args.get('act')
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register where uname=%s',(uname, ))
    rr=cursor1.fetchone()
    name=rr[1]

    cursor1.execute('SELECT * FROM dc_user_files where uname=%s',(uname, ))
    data=cursor1.fetchall()

    if act=="del":
        did = request.args.get('did')
        cursor1.execute('delete from dc_user_files where id=%s', (did,))
        mydb.commit()

    return render_template('view_files.html',msg=msg,name=name,data=data)


@app.route('/view_user', methods=['GET', 'POST'])
def view_user():
    msg=""
    act = request.args.get('act')
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor1 = mydb.cursor()
    cursor1.execute('SELECT * FROM dc_register where uname=%s',(uname, ))
    rr=cursor1.fetchone()
    name=rr[1]

    cursor1.execute('SELECT * FROM dc_user where owner=%s',(uname, ))
    data=cursor1.fetchall()

    if act=="del":
        did = request.args.get('did')
        cursor1.execute('delete from dc_user_files where id=%s', (did,))
        mydb.commit()

    return render_template('view_user.html',msg=msg,name=name,data=data)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    uname=""
    msg=""
    mess=""
    email=""
    act = request.args.get('act')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM dc_register where uname=%s",(uname,))
    value = mycursor.fetchone()
    dname=value[1]
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
        
    if request.method=='POST':
        
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        mobile=request.form['mobile']
        email=request.form['email']
        user=request.form['user']
        pass1=request.form['pass']
        location=request.form['location']
        desig=request.form['desig']
        

        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute('SELECT count(*) FROM dc_user WHERE uname = %s ', (user,))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            
            mycursor.execute("SELECT max(id)+1 FROM dc_user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            
            sql = "INSERT INTO dc_user(id, name, owner, gender, dob, mobile, email,location, desig, uname, pass) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, uname, gender, dob, mobile, email, location, desig, user, pass1)
            act="success"
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            ##send mail
            message="User Account - Data Owner:"+uname+", Username: "+user+", Password: "+pass1
            #url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
            #webbrowser.open_new(url)
            act="1"
            msg="ok"
        else:
            msg="fail"

    mycursor.execute("SELECT * FROM dc_user where owner=%s",(uname,))
    data = mycursor.fetchall()
    
    return render_template('add_user.html',value=value,act=act,data=data,dname=dname,msg=msg,mess=mess,email=email)



@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    act=""
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]

    
    cursor.execute("SELECT * FROM dc_user_files f,dc_share s where s.fid=f.id && s.uname=%s",(uname,))
    data = cursor.fetchall()

    return render_template('userhome.html',msg=msg,rr=rr,act=act,name=name,data=data)

@app.route('/file_verify', methods=['GET', 'POST'])
def file_verify():
    msg=""
    act=""
    fname = request.args.get('fname')
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]

    cursor.execute('SELECT * FROM dc_user_kgc where uname=%s',(uname, ))
    rr2=cursor.fetchone()
    pbkey2=rr2[5]
    
    cursor.execute('SELECT * FROM dc_register where uname=%s',(owner, ))
    rrd=cursor.fetchone()
    pbk=rrd[5]

    
    cursor.execute("SELECT * FROM dc_user_files where uname=%s",(owner,))
    data = cursor.fetchall()
    ###Decrypt by owner pbk#
    password_provided = pbk # This is input in the form of a string
    password = password_provided.encode() # Convert to type bytes
    salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    input_file = 'static/upload/e'+fname
    output_file = 'static/decrypted/'+fname
    with open(input_file, 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.decrypt(data)

    with open(output_file, 'wb') as f:
        f.write(encrypted)

    #######################################
    ##encrypt by user pbk
    password_provided = pbkey2 # This is input in the form of a string
    password = password_provided.encode() # Convert to type bytes
    salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))

    input_file = 'static/upload/'+fname
    output_file = 'static/encrypted/E'+fname
    with open(input_file, 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)

    with open(output_file, 'wb') as f:
        f.write(encrypted)
   
    ##############

        

            
    return render_template('file_verify.html',msg=msg,act=act,name=name,data=data,fname=fname,fid=fid)

@app.route('/access', methods=['GET', 'POST'])
def access():
    uname=""
    msg=""
    
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM dc_register where uname=%s",(uname,))
    value = mycursor.fetchone()

    mycursor.execute("SELECT * FROM dc_user where owner=%s",(uname,))
    data = mycursor.fetchall()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
        
    if request.method=='POST':
        
        uid=request.form.getlist('uid[]')
        #print(uid)
        for ss in uid:
            mycursor.execute("SELECT count(*) FROM dc_share where uname=%s && id=%s",(ss,fid))
            cnt = mycursor.fetchone()[0]
            if  cnt==0:
                mycursor.execute("SELECT max(id)+1 FROM dc_share")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1

                
                sql = "INSERT INTO dc_share(id, fid, uname, rdate) VALUES (%s, %s, %s, %s)"
                val = (maxid, fid, ss, rdate)
                act="success"
                mycursor.execute(sql, val)
                mydb.commit()
        return redirect(url_for('view_files'))


    mycursor.execute("SELECT * FROM dc_share where fid=%s",(fid,))
    data2 = mycursor.fetchall()
    
    return render_template('access.html',value=value,data=data,data2=data2)

@app.route('/file_down', methods=['GET', 'POST'])
def file_down():
    msg=""
    act=""
    fname = request.args.get('fname')
    fid = request.args.get('fid')
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()

    rn1=randint(1000,9999)
    ff=open("key.txt","w")
    ff.write(str(rn1))
    ff.close()
    
    data1="User: "+uname+", File Download, File: "+fname

    return render_template('file_down.html',fname=fname,fid=fid,bc=bc,data1=data1)

@app.route('/file_page', methods=['GET', 'POST'])
def file_page():
    msg=""
    data1=""
    fname = request.args.get('fname')
    fid = request.args.get('fid')
    act = request.args.get('act')
    if 'username' in session:
        uname = session['username']
    print(uname)

    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM dc_user where uname=%s',(uname, ))
    rr=cursor.fetchone()
    name=rr[1]
    owner=rr[2]
    mobile=rr[5]

    cursor.execute('SELECT * FROM dc_user_kgc where uname=%s',(uname, ))
    rr2=cursor.fetchone()
    prk=rr2[6]
    pbk=rr2[5]

    ff=open("bc.txt","r")
    bc=ff.read()
    ff.close()

    ff=open("key.txt","r")
    otp=ff.read()
    ff.close()

    
    st=""
    if request.method=='POST':
        skey=request.form['skey']
        if prk==skey:
            st="1"
            ###Decrypt by user pbk#
            password_provided = pbk # This is input in the form of a string
            password = password_provided.encode() # Convert to type bytes
            salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            input_file = 'static/encrypted/e'+fname
            output_file = 'static/decrypted/'+fname
            with open(input_file, 'rb') as f:
                data = f.read()

            fernet = Fernet(key)
            encrypted = fernet.decrypt(data)

            with open(output_file, 'wb') as f:
                f.write(encrypted)
            data1="User: "+uname+", File Download, File: "+fname 
        else:
            st="2"     
            data1="User: "+uname+", Attack found, File: "+fname

    return render_template('file_page.html',fname=fname,fid=fid,bc=bc,data1=data1,act=act,st=st,otp=otp,mobile=mobile)



@app.route('/down1', methods=['GET', 'POST'])
def down1():
    fn = request.args.get('fname')
    path="static/upload/"+fn
    return send_file(path, as_attachment=True)

@app.route('/down', methods=['GET', 'POST'])
def down():
    fn = request.args.get('fname')
    path="static/upload/"+fn
    return send_file(path, as_attachment=True)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
