using MaterialSkin.Controls;
using MaterialSkin;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Filosofiya
{
    public partial class Предпочтения : MaterialForm
    {
        
        public Предпочтения()
        {
            InitializeComponent();
            MaterialSkinManager materialSkinManager = MaterialSkinManager.Instance;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkinManager.Themes.LIGHT;
            
        }

        private void button1_Click(object sender, EventArgs e)
        {
            
        }

        private void webBrowser1_DocumentCompleted(object sender, WebBrowserDocumentCompletedEventArgs e)
        {
           
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            
            
        }

        private void Предпочтения_Load(object sender, EventArgs e)
        {
            checkBoxIdeal.Checked = Properties.Settings.Default.checkBoxIdeal;
            checkBoxMater.Checked = Properties.Settings.Default.checkBoxMater;
            checkBoxMiddleAges.Checked = Properties.Settings.Default.checkBoxMiddleAges;
            checkBoxNewTime.Checked = Properties.Settings.Default.checkBoxNewTime;
            checkBoxAntique.Checked = Properties.Settings.Default.checkBoxAntique;
        }

        private void Предпочтения_Validating(object sender, CancelEventArgs e)
        {
           
        }

        private void materialCheckBox1_CheckedChanged(object sender, EventArgs e)
        {
            
        }

        private void button1_Click_2(object sender, EventArgs e)
        {
            //все 5 / идеал матер / 3 эпохи
            if (checkBoxIdeal.Checked == true & checkBoxMater.Checked == true & checkBoxAntique.Checked == true & checkBoxMiddleAges.Checked == true & checkBoxNewTime.Checked == true | checkBoxIdeal.Checked == true & checkBoxMater.Checked == true & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false | checkBoxAntique.Checked == true & checkBoxMiddleAges.Checked == true & checkBoxNewTime.Checked == true & checkBoxIdeal.Checked == false & checkBoxMater.Checked == false)
            {
                Data.Предпочтения_значение = 1;
            }
            //Idealizm
            if (checkBoxIdeal.Checked == true & checkBoxMater.Checked == false & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 2;
            }
            //Materializm
            if (checkBoxMater.Checked == true & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 3;
            }
            //Antique
            if (checkBoxAntique.Checked == true & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 4;
            }
            //Middle ages
            if (checkBoxMiddleAges.Checked == true & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 5;
            }
            //Новое время
            if (checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 6;
            }
            //Ideal + antique
            if (checkBoxIdeal.Checked & checkBoxAntique.Checked & checkBoxMater.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 7;
            }
            //Ideal + Middle Ages
            if (checkBoxIdeal.Checked & checkBoxMiddleAges.Checked & checkBoxMater.Checked == false & checkBoxAntique.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 8;
            }
            //Ideal + new time
            if (checkBoxIdeal.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 9;
            }
            //Mater + antique
            if (checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxIdeal.Checked == false & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 10;
            }
            //Mater + Middle Ages
            if (checkBoxMater.Checked & checkBoxMiddleAges.Checked & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 11;
            }
            //Mater + new time
            if (checkBoxMater.Checked & checkBoxNewTime.Checked & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 12;
            }
            //Antique + Middle ages
            if (checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 13;
            }
            //Antique + new time
            if (checkBoxAntique.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 14;
            }
            //Middle ages + new time
            if (checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false)
            {
                Data.Предпочтения_значение = 15;
            }
            //Ideal mater antique
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxMiddleAges.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 16;
            }
            //Ideal mater middle age
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxMiddleAges.Checked & checkBoxAntique.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 17;
            }
            //Ideal mater new time
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxNewTime.Checked & checkBoxAntique.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 18;
            }
            //ideal antique middle age
            if (checkBoxIdeal.Checked & checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxMater.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 19;
            }
            //ideal middle age new time
            if (checkBoxIdeal.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxAntique.Checked == false)
            {
                Data.Предпочтения_значение = 20;
            }
            //ideal antique new time
            if (checkBoxIdeal.Checked & checkBoxAntique.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 21;
            }
            //mater antique middle age
            if (checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxIdeal.Checked == false & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 22;
            }
            //mater middle age new time
            if (checkBoxMater.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxIdeal.Checked == false & checkBoxAntique.Checked == false)
            {
                Data.Предпочтения_значение = 23;
            }
            //mater antique new time
            if (checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxNewTime.Checked & checkBoxIdeal.Checked == false & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 24;
            }
            //ideal mater antique middle age
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked == false)
            {
                Data.Предпочтения_значение = 25;
            }
            //ideal mater antique new time 
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxAntique.Checked & checkBoxNewTime.Checked & checkBoxMiddleAges.Checked == false)
            {
                Data.Предпочтения_значение = 26;
            }
            //ideal mater middle ages new time 
            if (checkBoxIdeal.Checked & checkBoxMater.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxAntique.Checked == false)
            {
                Data.Предпочтения_значение = 27;
            }
            //3 эпохи + матер
            if (checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxMater.Checked & checkBoxIdeal.Checked == false)
            {
                Data.Предпочтения_значение = 28;
            }
            //3 эпохи + Idealizm
            if (checkBoxAntique.Checked & checkBoxMiddleAges.Checked & checkBoxNewTime.Checked & checkBoxIdeal.Checked & checkBoxMater.Checked == false)
            {
                Data.Предпочтения_значение = 29;
            }
            основное1 f2 = new основное1();
            this.Hide();
            Предпочтения f1 = new Предпочтения();
            f1.Close();
            f2.Show();


            Properties.Settings.Default.checkBoxIdeal = checkBoxIdeal.Checked;
            Properties.Settings.Default.checkBoxMater = checkBoxMater.Checked;
            Properties.Settings.Default.checkBoxMiddleAges = checkBoxMiddleAges.Checked;
            Properties.Settings.Default.checkBoxNewTime = checkBoxNewTime.Checked;
            Properties.Settings.Default.checkBoxAntique = checkBoxAntique.Checked;
            Properties.Settings.Default.Save();
        }

        private void Предпочтения_FormClosed(object sender, FormClosedEventArgs e)
        {
            //checkBoxIdeal
            //checkBoxMater
            //checkBoxAntique
            //checkBoxMiddleAges
            //checkBoxNewTime
            Properties.Settings.Default.checkBoxIdeal = checkBoxIdeal.Checked;
            Properties.Settings.Default.checkBoxMater = checkBoxMater.Checked;
            Properties.Settings.Default.checkBoxMiddleAges = checkBoxMiddleAges.Checked;
            Properties.Settings.Default.checkBoxNewTime = checkBoxNewTime.Checked;
            Properties.Settings.Default.checkBoxAntique = checkBoxAntique.Checked;
            Properties.Settings.Default.Save();

        }
    }
}
