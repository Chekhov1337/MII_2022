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
    public partial class Браузер_Form3_ : Form
    {
       
        public Браузер_Form3_()
        {
            if (основное1.автор != null && Data.Узнать_про_цитату_и_автора == 1)
            {
                InitializeComponent();
                Uri url = new Uri("https://yandex.ru/images/search?text=" + основное1.автор);
                webControl1.Source = url;
            }
            if(основное1.цитата != null && Data.Узнать_про_цитату_и_автора == 2)
            {
                InitializeComponent();
                Uri url = new Uri("https://yandex.ru/search/?text=" + основное1.цитата);
                webControl1.Source = url;
            }
        }


        private void Awesomium_Windows_Forms_WebControl_ShowCreatedWebView_1(object sender, Awesomium.Core.ShowCreatedWebViewEventArgs e)
        {
            
        }
    }
}
