using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Net;
using System.Diagnostics;
using Tulpep.NotificationWindow;
using MaterialSkin.Controls;
using MaterialSkin;
using System.Threading;
using System.Timers;
using System.Net.Sockets;
using System.Net.NetworkInformation;
using System.Net.Http;
namespace Filosofiya
{
    
    public partial class основное1 : MaterialForm
    {
        
        private PopupNotifier popup = null;
        string line; // line = mas
        string[] mas; // массив-вывод
        public static string цитата; // основной вывод 
        public static string автор;
        int p;
        private NotifyIcon NI = new NotifyIcon(); // уведомления 2 
        public основное1()
        {
            InitializeComponent();
            label2.Text = "";
            label2.Hide();
            timer1.Enabled = true;
            timer1.Interval = 1000;
        }
        
        private void label1_Click(object sender, EventArgs e)
        {
            
        }

        public void Шрифт()
        {
            //label1.Font.Size = 200;
        }
        public void Массив(string line)
        {
            if (Data.Выдача_цитат == 0)
            {
                //.Remove(цитата.IndexOf(" (")
                mas = line.Split('\n');
                Random rand = new Random();
                int num = rand.Next(0, mas.Length);
                цитата = mas[num];
                Console.WriteLine(цитата.Remove(цитата.IndexOf(" (")));
                label1.Text = цитата.Remove(цитата.IndexOf(" ("));
            }
            if (Data.Выдача_цитат == 1)
            {
                mas = line.Split('\n');
                Random rand = new Random();
                int num = rand.Next(0, mas.Length);
                цитата = mas[num];
                Console.WriteLine(цитата.Remove(цитата.IndexOf(" (")));
                label1.Text = цитата.Remove(цитата.IndexOf(" ("));
            }
            if (Data.Выдача_цитат == 2)
            {
                //Последовательный вывод 
                mas = line.Split('\n');
                if (p >= mas.Length)
                {
                    return;
                }
                else
                {
                    цитата = mas[p];
                    Console.WriteLine(цитата.Remove(цитата.IndexOf(" (")));
                    label1.Text = цитата.Remove(цитата.IndexOf(" ("));
                }
            }
        }
        public void Рандом(string[] mas, string line)
        {
            //Data.Выдача_цитат = 1;
            // считываем Sample.txt
            if (Data.Предпочтения_значение == 1) //Все
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd() + "\n" + sr3.ReadToEnd() + "\n" + sr4.ReadToEnd() + "\n" + sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr1.Close();
                sr2.Close();
                sr3.Close();
                sr4.Close();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 2) //Идеализм
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                line = sr1.ReadToEnd() + "\n" + sr3.ReadToEnd() + "\n" + sr5.ReadToEnd();
                sr1.Close();
                sr3.Close();
                sr5.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 3) //Материализм
            {
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr2.ReadToEnd() + "\n" + sr4.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr2.Close();
                sr4.Close();
                sr6.Close();
                Массив(line);

            }
            if (Data.Предпочтения_значение == 4) //Античные челы
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd();
                sr1.Close();
                sr2.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 5) //Средние века
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr3.ReadToEnd() + "\n" + sr4.ReadToEnd();
                sr3.Close();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 6) //Новое время
            {
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 7) //Античные идеалисты
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                line = sr1.ReadToEnd();
                sr1.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 8) //Идеалисты средних веков
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                line = sr3.ReadToEnd();
                sr3.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 9) //Идеалисты нового времени
            {
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                line = sr5.ReadToEnd();
                sr5.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 10) //Материалисты античности
            {
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                line = sr2.ReadToEnd();
                sr2.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 11) //Материалисты Ср.Веков
            {
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr4.ReadToEnd();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 12) //Материалисты нового времени
            {
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr6.ReadToEnd();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 13) //Античность и ср. века
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd() + "\n" + sr3.ReadToEnd() + "\n" + sr4.ReadToEnd();
                sr1.Close();
                sr2.Close();
                sr3.Close();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 14) //Античность + новое время
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd() + "\n" + sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr1.Close();
                sr2.Close();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 15) //Ср. Века + Новое Время
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr3.ReadToEnd() + "\n" + sr4.ReadToEnd() + "\n" + sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr3.Close();
                sr4.Close();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 16) //Идеалисты материалисты античности
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd();
                sr1.Close();
                sr2.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 17) //Идеалисты материалисты Ср.Веков
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr3.ReadToEnd() + "\n" + sr4.ReadToEnd();
                sr3.Close();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 18) //Идеалисты материалисты Нового Времени
            {
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 19) //Идеалисты античности и ср. веков
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                line = sr1.ReadToEnd() + "\n" + sr3.ReadToEnd();
                sr1.Close();
                sr3.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 20) //Идеалисты ср. веков и Нового времени
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                line = sr3.ReadToEnd() + "\n" + sr5.ReadToEnd();
                sr3.Close();
                sr5.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 21) //Идеалисты аничности и нового времнеи
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                line = sr1.ReadToEnd() + "\n" + sr5.ReadToEnd();
                sr1.Close();
                sr5.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 22) //Материалисты античности и ср. веков
            {
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr2.ReadToEnd() + "\n" + sr4.ReadToEnd();
                sr2.Close();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 23) //Материалисты ср. веков и нового времени
            {
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr4.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr4.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 24) //Материалисты античности и нового времени
            {
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr2.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr2.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 25) //Идеалисты и материалисты античности и СР.Веков
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd() + "\n" + sr3.ReadToEnd() + "\n" + sr4.ReadToEnd();
                sr1.Close();
                sr2.Close();
                sr3.Close();
                sr4.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 26) //Идеалисты и материалисты античности и нового времени
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr1.ReadToEnd() + "\n" + sr2.ReadToEnd() + "\n" + sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr1.Close();
                sr2.Close();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 27) //Идеалисты и материалисты ср веков и нового времени
            {
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr3.ReadToEnd() + "\n" + sr4.ReadToEnd() + "\n" + sr5.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr3.Close();
                sr4.Close();
                sr5.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 28) //3 Эпохи + материализм
            {
                StreamReader sr2 = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt");
                StreamReader sr4 = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt");
                StreamReader sr6 = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt");
                line = sr2.ReadToEnd() + "\n" + sr4.ReadToEnd() + "\n" + sr6.ReadToEnd();
                sr2.Close();
                sr4.Close();
                sr6.Close();
                Массив(line);
            }
            if (Data.Предпочтения_значение == 29) //3 Эпохи + Идеализм
            {
                StreamReader sr1 = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt");
                StreamReader sr3 = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt");
                StreamReader sr5 = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt");
                line = sr1.ReadToEnd() + "\n" + sr3.ReadToEnd() + "\n" + sr5.ReadToEnd();
                sr1.Close();
                sr3.Close();
                sr5.Close();
                Массив(line);
            }
        }
        public void Автор()
        {
            try
            {
                string test = цитата;
                string te = test.Substring(test.LastIndexOf("(") + 1);
                автор = te.Substring(0, te.LastIndexOf(")"));
                linkLabel1.Text = автор;
            }
            catch
            {
                var request = (HttpWebRequest)WebRequest.Create("http://g.cn/generate_204");
                request.UserAgent = "Android";
                request.KeepAlive = false;
                request.Timeout = 1500;
                using (var response = (HttpWebResponse)request.GetResponse())
                {

                    if (response.ContentLength == 0 && response.StatusCode == HttpStatusCode.NoContent)
                    {
                        MessageBox.Show("Почему-то не удается загрузить цитаты...\n" + "Пытаюсь исправить");
                        //AntiqueIdeal
                        WebClient wc = new WebClient();
                        string url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/AntiqueIdeal.txt";
                        string save_path = @".\Resources\цитаты\";
                        string name = "AntiqueIdeal.txt";
                        wc.DownloadFile(url, save_path + name);
                        //AntiqueMater
                        WebClient wc2 = new WebClient();
                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/AntiqueMater.txt";
                        string save_path2 = @".\Resources\цитаты\";
                        string name2 = "AntiqueMater.txt";
                        wc2.DownloadFile(url, save_path2 + name2);
                        //MiddleAgesIdeal
                        WebClient wc3 = new WebClient();
                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/MiddleAgesIdeal.txt";
                        string save_path3 = @".\Resources\цитаты\";
                        string name3 = "MiddleAgesIdeal.txt";
                        wc3.DownloadFile(url, save_path3 + name3);
                        //MiddleAgesMater
                        WebClient wc4 = new WebClient();
                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/MiddleAgesMater.txt";
                        string save_path4 = @".\Resources\цитаты\";
                        string name4 = "MiddleAgesMater.txt";
                        wc3.DownloadFile(url, save_path4 + name4);
                        //NewTimeIdeal
                        WebClient wc5 = new WebClient();
                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/NewTimeIdeal.txt";
                        string save_path5 = @".\Resources\цитаты\";
                        string name5 = "NewTimeIdeal.txt";
                        wc3.DownloadFile(url, save_path5 + name5);
                        //NewTimeMater
                        WebClient wc6 = new WebClient();
                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/NewTimeMater.txt";
                        string save_path6 = @".\Resources\цитаты\";
                        string name6 = "NewTimeMater.txt";
                        wc3.DownloadFile(url, save_path6 + name6);
                        MessageBox.Show("Файлы восстановлены:" + name + " " + name2 + " " + name3 + " " + name4 + " " + name5 + " " + name6);

                    }
                }
                
            }
        }
        public void Узнать_об_авторе()
        {
            if (автор != null)
            {
                Data.Узнать_про_цитату_и_автора = 1;
                Браузер_Form3_ f3 = new Браузер_Form3_();
                f3.Show();
            }
            else
            {
                MessageBox.Show("Нужно узнать автора!");
            }
        }
        public void Узнать_про_цитату()
        {
            if (цитата != null)
            {
                Data.Узнать_про_цитату_и_автора = 2;
                Браузер_Form3_ f3 = new Браузер_Form3_();
                f3.Show();
            }
            else
            {
                MessageBox.Show("Нужно вывести цитату!");
            }
        }
        public void Получение_изображение_об_авторе()
        {
            try
            {
                pictureBox1.Image = null;
                pictureBox1.Image = Image.FromFile(@".\Resources\img\" + автор.ToString() + ".jpg");
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            }
            catch
            {
                pictureBox1.Image = null;
                pictureBox1.Image = Image.FromFile(@".\Resources\img\noname.jpg");
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            }
        }

        public void Уведомления(string цитата)
        {
            popup = new PopupNotifier();
            //popup.Image = Properties.Resources.Image1;
            popup.ImageSize = new Size(96, 96); // размер изображения
            //popup.TitleText = цитата;
            popup.HeaderColor = Color.White; // цвет верхушки
            popup.BodyColor = Color.White;  // основной цвет
            popup.AnimationInterval = 20; // скорость анимации
            popup.ContentText = цитата;
            popup.ContentFont = new Font("Times New Roman", 12); // цвет заголовка
            popup.TitleFont = new Font("Times New Roman", 12); // цвет основн. текста
            popup.TitleColor = Color.Black; //цвет текста
            popup.Popup();
        }
        public void Все_в_месте()
        {
            Рандом(mas, line);
            if (toolStripComboBox1.SelectedIndex == 0)
            {
                Уведомления(цитата.Remove(цитата.IndexOf(" (")));
            }
            else if (toolStripComboBox1.SelectedIndex == 1)
            {
                Notifier();
            }
            else if (toolStripComboBox1.SelectedIndex == 2)
            {

            }
            Получение_изображение_об_авторе();
        }
        private void button1_Click(object sender, EventArgs e)
        {
            Все_в_месте();
            if (Data.Выдача_цитат == 2)
            { 
                p++; 
            }
        }

        private void label1_TextChanged(object sender, EventArgs e)
        {
            Автор();
        }
        

        private void основное1_Load(object sender, EventArgs e)
        {
            
           
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            Узнать_об_авторе();

        }
        public void Notifier()
        {
            Рандом(mas, line);
            NI.BalloonTipText = цитата;
            NI.BalloonTipTitle = "";
            NI.BalloonTipIcon = ToolTipIcon.None;
            NI.Icon = new Icon(@".\philosophy.ico");
            NI.Visible = true;
            NI.ShowBalloonTip(1000);
        }
        private void NI_BallonTipClosed(Object sender, EventArgs e)
        {
            NI.Visible = false;
        }

        private void toolStripComboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            var materialskin = MaterialSkinManager.Instance;
            materialskin.AddFormToManage(this);
            if (toolStripComboBox2.SelectedIndex == 0)
            {
                materialskin.Theme = MaterialSkinManager.Themes.DARK;
                this.Controls["label1"].BackColor = Color.FromName("Gray");
                this.Controls["linkLabel1"].BackColor = Color.FromName("Gray");
                this.Controls["linkLabel2"].BackColor = Color.FromName("Gray");
                this.Controls["menuStrip1"].BackColor = Color.FromName("Gray");
                this.Controls["button1"].BackColor = Color.FromName("Gray");
                this.Controls["button1"].ForeColor = Color.FromName("ControlDark");

            }
            else if (toolStripComboBox2.SelectedIndex == 1)
            {
                
                materialskin.Theme = MaterialSkinManager.Themes.LIGHT;
                this.Controls["label1"].BackColor = Color.FromName("White");
                this.Controls["linkLabel1"].BackColor = Color.FromName("White");
                this.Controls["linkLabel2"].BackColor = Color.FromName("White");
                this.Controls["menuStrip1"].BackColor = Color.FromName("White");
                this.Controls["button1"].BackColor = Color.FromName("Control");
                this.Controls["button1"].ForeColor = Color.FromName("ControlText");

            }
        }

        private void выходToolStripMenuItem_Click(object sender, EventArgs e)
        {

            Application.Exit();
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            
            label2.Text = DateTime.Now.ToLongTimeString();
        }

        private void toolStripComboBox3_TextChanged(object sender, EventArgs e)
        {
            if (toolStripComboBox3.SelectedIndex == 0)
            {
                label2.Show();
            }
            else if (toolStripComboBox3.SelectedIndex == 1)
            {
                label2.Hide();
            }
        }

        private void timer2_Tick(object sender, EventArgs e)
        {
            Рандом(mas, line);
            if (toolStripComboBox1.SelectedIndex == 0)
            {
                Уведомления(цитата);
            }
            else if (toolStripComboBox1.SelectedIndex == 1)
            {
                Notifier();
            }
            else if (toolStripComboBox1.SelectedIndex == 2)
            {

            }
            Получение_изображение_об_авторе();
        }

        private void toolStripComboBox4_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (toolStripComboBox4.SelectedIndex == 0 )
            {
                label2.Show();
                timer2.Interval = Convert.ToInt32(Data.Value);
                timer2.Start();
                button1.Hide();
            }
           else if (toolStripComboBox4.SelectedIndex == 1)
            {
                label2.Hide();
                timer2.Stop();
                button1.Show();
            }    
        }
        // открываем текущее настройки
        int a = 4;
        int b = 2;
        int c = 0;
        int d = 1;
        public void записатьЦитатуToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Data.Вкладка = 3;
            Настройки form4 = new Настройки(this.a);
            form4.ShowDialog();
        }

        private void таймерToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Data.Вкладка = 4;
            Настройки form4 = new Настройки(this.b);
            form4.Show();
        }

        private void ТаймерToolStripMenuItem_Click(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private void настройкиШрифтаЧасовToolStripMenuItem_Click(object sender, EventArgs e)
        {
            fontDialog1.ShowDialog();
            label2.Font = fontDialog1.Font;
        }

        private void настройкиШрифтаЦитатыToolStripMenuItem_Click(object sender, EventArgs e)
        {
            fontDialog1.ShowDialog();
            label1.Font = fontDialog1.Font;
        }

        private void выводЦитатыToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Data.Вкладка = 1;
            Настройки form4 = new Настройки(this.c);
            form4.Show();
           
            
        }

        private void обновлениеЦитатыToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Data.Вкладка = 2;
            Настройки form4 = new Настройки(this.d);
            form4.Show();
        }

        private void toolStripComboBox2_Click(object sender, EventArgs e)
        {

        }

        private void materialRaisedButton1_Click(object sender, EventArgs e)
        {
            Все_в_месте();
            if (Data.Выдача_цитат == 2)
            {
                p++;
            }
        }

        private void label1_Click_1(object sender, EventArgs e)
        {
            Узнать_про_цитату();
        }

        private void предпочтенияToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Предпочтения form1 = new Предпочтения();
            form1.Show();
        }

        private void автозагрузкаToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Data.Вкладка = 10;
            Настройки form4 = new Настройки(this.d);
            form4.Show();
        }
        
        private void linkLabel2_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            int b;
            Data.Вкладка = 14;
            Настройки form4 = new Настройки(this.b);
            form4.Show();
        }
    }
}


