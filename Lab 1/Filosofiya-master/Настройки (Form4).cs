using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using MaterialSkin.Controls;
using MaterialSkin;
using System.Net;
using System.Net.Sockets;
using System.Net.NetworkInformation;
using System.Net.Http;
using System.Threading;
using Microsoft.Win32;
using System.Diagnostics;
namespace Filosofiya
{
    public partial class Настройки : MaterialForm
    {
        public void Считываем()
        {
//AntiqueIdeal
//AntiqueMater
//MiddleAgesIdeal
//MiddleAgesMater
//NewTimeIdeal
//NewTimeMater
            try
            {
                if (comboBox2.SelectedIndex == -1)
                {
                    label2.Text = "Воспользуйтесь выпадающим списком! (combobox)";
                }
                if (comboBox2.SelectedIndex == 0) //Материализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\AntiqueIdeal.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
                if (comboBox2.SelectedIndex == 1) //Идеализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\AntiqueMater.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
                if (comboBox2.SelectedIndex == 2) //Идеализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\MiddleAgesIdeal.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
                if (comboBox2.SelectedIndex == 3) //Идеализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\MiddleAgesMater.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
                if (comboBox2.SelectedIndex == 4) //Идеализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\NewTimeIdeal.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
                if (comboBox2.SelectedIndex == 5) //Идеализм
                {
                    label2.Hide();
                    using (StreamReader myReader = new StreamReader(@".\Resources\цитаты\NewTimeMater.txt", Encoding.UTF8))
                    {
                        richTextBox1.Text = myReader.ReadToEnd() /*+ "\r\n"*/;
                    }
                }
            }
            catch
            {
                MessageBox.Show("Что-то случилось");
            }
        }
        public void Запись()
        {
            
            try
            {
                if (comboBox2.SelectedIndex == 0)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\AntiqueIdeal.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
                if (comboBox2.SelectedIndex == 1)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\AntiqueMater.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
                if (comboBox2.SelectedIndex == 2)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\MiddleAgesIdeal.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
                if (comboBox2.SelectedIndex == 3)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\MiddleAgesMater.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
                if (comboBox2.SelectedIndex == 4)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\NewTimeIdeal.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
                if (comboBox2.SelectedIndex == 5)
                {
                    using (StreamWriter myWriter = new StreamWriter(@".\Resources\цитаты\NewTimeMater.txt"))
                    {
                        string line = richTextBox1.Text;
                        myWriter.Write(line);
                        label2.Text = "Записал!";
                    }
                }
            }
            catch
            {
                MessageBox.Show("Что-то случилось");
            }
        }
        int data;
        public Настройки(int data)
        {
            InitializeComponent();
            Считываем();
            this.data = data;
        }
        string[] mas2;
        string line2;
        public void Источники()
        {
            button12.Hide();
            textBox3.Hide();
            StreamReader sr1 = new StreamReader(@".\Resources\links.txt");
            string line2 = sr1.ReadToEnd();
            sr1.Close();
            if (line2 != null)
            {
                mas2 = line2.Split('\n');
                for (int i = 0; i < mas2.Length; i++)
                {
                    listBox2.Items.Add(mas2[i]);
                }
            }
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            Источники();
            textBox2.Hide();
            button7.Hide();
            if (data == 4 || Data.Вкладка == 3 )
            {
                tabControl1.SelectedIndex = 0;
            }
            if (data == 2 || Data.Вкладка == 4)
            {
                tabControl1.SelectedIndex = 1;
            }
            if (Data.Вкладка == 1)
            {
                tabControl1.SelectedIndex = 2;
            }
            if (Data.Вкладка == 2)
            {
                tabControl1.SelectedIndex = 3;
            }
            if (Data.Вкладка == 10)
            {
                tabControl1.SelectedIndex = 4;
            }
            if (Data.Вкладка == 14)
            {
                tabControl1.SelectedIndex = 5;
            }
        }

        private void tabPage1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Запись();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Считываем();
        }

        private void tabPage2_Click(object sender, EventArgs e)
        {

        }
        private void button3_Click(object sender, EventArgs e)
        {
           float time_set = Convert.ToInt32(textBox1.Text);
            // проверка на ошибки
            if (comboBox1.SelectedIndex == -1)
            {
                label1.Text = "Что-то пошло не так";
            }
            //в часах
            if (comboBox1.SelectedIndex == 0)
            {
                Data.Value = time_set * 3600000;
                label1.Text = "Настройки применены";
            }
            //в минутах
            if (comboBox1.SelectedIndex == 1)
            {
                Data.Value = time_set * 60000;
                label1.Text = "Настройки применены";
            }
            //секунды
            if (comboBox1.SelectedIndex == 2)
            {
                Data.Value = time_set * 1000;
                label1.Text = "Настройки применены";
            }
            if (comboBox1.SelectedIndex == 3)
            {
                Data.Value = time_set;
                label1.Text = "Настройки применены";
            }
        }
        string url;
        string url2;
        public void proverka_url (string url2)
        {
            try
            {
                if (!string.IsNullOrEmpty(url2))
                {
                    UriBuilder uriBuilder = new UriBuilder(url2);
                    HttpWebRequest request = (HttpWebRequest)HttpWebRequest.Create(uriBuilder.Uri);
                    HttpWebResponse response = (HttpWebResponse)request.GetResponse();
                    if (response.StatusCode == HttpStatusCode.NotFound)
                    {
                        MessageBox.Show("Данный url не подходит");
                    }
                    if (response.StatusCode == HttpStatusCode.OK)
                    { 
                        MessageBox.Show("Успешно!");
                    }
                    else //There are a lot of other status codes you could check for...
                    {
                        MessageBox.Show(string.Format("Данный url не подходит.",
                                                   response.StatusCode.ToString()));
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(string.Format("Broken- Other error: {0}", ex.Message));
            }
        }
        public void ping()
        {
            if (listBox1.SelectedIndex == -1)
            {
                MessageBox.Show("Воспользуйтесь списком");
            }
            else
            {
                var request = (HttpWebRequest)WebRequest.Create("http://g.cn/generate_204");
                request.UserAgent = "Android";
                request.KeepAlive = false;
                request.Timeout = 1500;

                try
                {
                    label4.Text = "Проверяем доступ к  интернету...";
                    Thread.Sleep(1500);
                    progressBar1.Value = 3;
                    using (var response = (HttpWebResponse)request.GetResponse())
                    {
                        if (response.ContentLength == 0 && response.StatusCode == HttpStatusCode.NoContent)
                        {

                            progressBar1.Value = 10;
                            int a = 15;
                            if (progressBar1.Value < a)
                            {
                                label4.Text = "Происходит обновление цитаты..";
                                Thread.Sleep(1500);
                                WebClient wc = new WebClient();
                                //AntiqueIdea
                                if (listBox1.SelectedIndex == 0)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/AntiqueIdeal.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "AntiqueIdea.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }
                                //AntiqueMater
                                if (listBox1.SelectedIndex == 1)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/AntiqueMater.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "AntiqueMater.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }
                                //MiddleAgesIdeal
                                if (listBox1.SelectedIndex == 2)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/MiddleAgesIdeal.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "MiddleAgesIdeal.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }
                                //MiddleAgesMater
                                if (listBox1.SelectedIndex == 3)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/MiddleAgesMater.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "MiddleAgesMater.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }
                                //NewTimeIdeal
                                if (listBox1.SelectedIndex == 4)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/NewTimeIdeal.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "NewTimeIdeal.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }
                                //NewTimeMater
                                if (listBox1.SelectedIndex == 5)
                                {
                                    if (button_1 == 0)
                                    {
                                        url = "https://raw.githubusercontent.com/Letov1337/Filosofiya/master/Resources/%D1%86%D0%B8%D1%82%D0%B0%D1%82%D1%8B/NewTimeMater.txt";
                                    }
                                    else if (button_1 == 1)
                                    {
                                        url = url2;
                                        button_1 = button_1 - 1;
                                    }
                                    string save_path = @".\Resources\цитаты\";
                                    string name = "NewTimeMater.txt";
                                    wc.DownloadFile(url, save_path + name);
                                    progressBar1.Value = 20;
                                    label4.Text = "Успешно обновлено";
                                    MessageBox.Show("Успешно обновлено:" + name);
                                }

                            }
                        }
                        else
                        {
                            MessageBox.Show("not ok");
                            progressBar1.Value = 0;
                        }
                    }
                }
                catch
                {
                    MessageBox.Show("not ok");
                    progressBar1.Value = 0;
                }
            }
        }
        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void button4_Click(object sender, EventArgs e)
        {
            
        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            Считываем();
            
        }

        private void tabPage3_Click(object sender, EventArgs e)
        {

        }

        private void button4_Click_1(object sender, EventArgs e)
        {
            if (comboBox3.SelectedIndex == 0)
            {
                Data.Выдача_цитат = 1; // Случайно
            }
            if (comboBox3.SelectedIndex == 1)
            {
                Data.Выдача_цитат = 2; // Рандомно
            }
        }
        private void button5_Click(object sender, EventArgs e)
        {
            ping();
        }
        int button_1 = 0; // Если у нас сработала кнопка
        
        
        private void button7_Click(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex == -1)
            {
                MessageBox.Show("Выберете из списка!");
            }
            else
            {
                url2 = textBox2.Text;
                proverka_url(url2);
                button_1 = 1;
            }
        }
        int button_2 = 0;// Если у нас сработала кнопка
        private void button6_Click(object sender, EventArgs e)
        {
            
            if (button_2 == 0)
            {
                button_2 = button_2 + 1;
                textBox2.Show();
                button7.Show();
            }
            else
            {
                button_2 = button_2 - 1;
                textBox2.Hide();
                button7.Hide();
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            if (comboBox4.SelectedIndex == -1)
            {
                MessageBox.Show("Нужно что-то выбрать");
            }
            if (comboBox4.SelectedIndex == 0)
            {
                    const string applicationName = "Quote Generator";
                    const string pathRegistryKeyStartup =
                                "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run";

                    using (RegistryKey registryKeyStartup =
                                Registry.CurrentUser.OpenSubKey(pathRegistryKeyStartup, true))
                    {
                        registryKeyStartup.DeleteValue(applicationName, false);
                    }
                    MessageBox.Show("Автозагрузка выключена!");
            }
            if (comboBox4.SelectedIndex == 1)
            {
                const string applicationName = "Quote Generator";
                const string pathRegistryKeyStartup =
                            "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run";

                using (RegistryKey registryKeyStartup =
                            Registry.CurrentUser.OpenSubKey(pathRegistryKeyStartup, true))
                {
                    registryKeyStartup.SetValue(
                        applicationName,
                        string.Format("\"{0}\"", System.Reflection.Assembly.GetExecutingAssembly().Location));
                }
                MessageBox.Show("Автозагрузка включена!");
            }
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void button9_Click(object sender, EventArgs e)
        {
            string sourceFile = @".\Resources\img\";
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            openFileDialog1.Filter = "Images files (*.jpg)|*.jpg|All files (*.*)|*.*";
            if (openFileDialog1.ShowDialog() == DialogResult.Cancel)
            {
                return;
            }
            // получаем выбранный файл
            string filename = openFileDialog1.FileName;
            //откуда копируем
            string Dir1 = filename;
            //куда копируем
            string Dir2 = sourceFile;
            try
            {
                DirectoryInfo dirInfo = new DirectoryInfo(Dir1);
                foreach (FileInfo file in dirInfo.GetFiles("*.*"))
                {
                    File.Copy(file.FullName, Dir2 + "\\" + file.Name, true);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        
        private void button11_Click(object sender, EventArgs e)
        {
            string copy = listBox2.SelectedItem.ToString();
            Clipboard.SetText(copy);
            MessageBox.Show("Скопировано в буфер обмена:\n" + copy);
        }

        private void button10_Click(object sender, EventArgs e)
        {
            textBox3.Show();
            button12.Show();
        }

        private void button12_Click(object sender, EventArgs e)
        {
            //using (StreamWriter myWriter = new StreamWriter(@".\Resources\links.txt"))
            //{
            //string line4 = textBox3.Text;
            //myWriter.Write(line4 + Environment.NewLine);
            //}
            //StreamReader sr1 = new StreamReader(@".\Resources\links.txt");
            //line2 = sr1.ReadToEnd();
            //sr1.Close();
            //mas2 = line2.Split('\n');
            //foreach(string listline in mas2)
            //{
            //    listBox2.Items.Add(listline);
            //}
            string line4 = textBox3.Text;
            listBox2.Items.Add(line4);
            using (StreamWriter myWriter = new StreamWriter(@".\Resources\links.txt"))
            {
               foreach(var item in listBox2.Items)
                {
                    myWriter.WriteLine(item);
                }
            }
            textBox3.Hide();
            button12.Hide();
        }
    }
}
