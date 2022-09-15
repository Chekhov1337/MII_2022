
namespace Filosofiya
{
    partial class основное1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(основное1));
            this.label1 = new System.Windows.Forms.Label();
            this.linkLabel1 = new System.Windows.Forms.LinkLabel();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.label2 = new System.Windows.Forms.Label();
            this.timer2 = new System.Windows.Forms.Timer(this.components);
            this.fontDialog1 = new System.Windows.Forms.FontDialog();
            this.button1 = new MaterialSkin.Controls.MaterialRaisedButton();
            this.linkLabel2 = new System.Windows.Forms.LinkLabel();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.настройкиToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.уведомленияToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripComboBox1 = new System.Windows.Forms.ToolStripComboBox();
            this.типToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripComboBox3 = new System.Windows.Forms.ToolStripComboBox();
            this.таймерToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripComboBox4 = new System.Windows.Forms.ToolStripComboBox();
            this.автозагрузкаToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.выводЦитатыToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.записатьЦитатуToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.обновлениеЦитатыToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.темаToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripComboBox2 = new System.Windows.Forms.ToolStripComboBox();
            this.настройкиШрифтаToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.настройкиШрифтаЧасовToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.настройкиШрифтаЦитатыToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.предпочтенияToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.выходToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12.25F);
            this.label1.ForeColor = System.Drawing.SystemColors.ControlText;
            this.label1.Location = new System.Drawing.Point(163, 82);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(515, 119);
            this.label1.TabIndex = 0;
            this.label1.Text = "Цитата:";
            this.label1.TextChanged += new System.EventHandler(this.label1_TextChanged);
            this.label1.Click += new System.EventHandler(this.label1_Click_1);
            // 
            // linkLabel1
            // 
            this.linkLabel1.AutoSize = true;
            this.linkLabel1.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.linkLabel1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12.25F);
            this.linkLabel1.LinkColor = System.Drawing.Color.Black;
            this.linkLabel1.Location = new System.Drawing.Point(163, 201);
            this.linkLabel1.Name = "linkLabel1";
            this.linkLabel1.Size = new System.Drawing.Size(61, 20);
            this.linkLabel1.TabIndex = 2;
            this.linkLabel1.TabStop = true;
            this.linkLabel1.Text = "Автор";
            this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel1_LinkClicked);
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.SystemColors.ButtonHighlight;
            this.label2.Font = new System.Drawing.Font("Comic Sans MS", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label2.Location = new System.Drawing.Point(622, 292);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(56, 23);
            this.label2.TabIndex = 6;
            this.label2.Text = "label2";
            // 
            // timer2
            // 
            this.timer2.Interval = 60000;
            this.timer2.Tick += new System.EventHandler(this.timer2_Tick);
            // 
            // button1
            // 
            this.button1.Depth = 0;
            this.button1.Location = new System.Drawing.Point(163, 227);
            this.button1.MouseState = MaterialSkin.MouseState.HOVER;
            this.button1.Name = "button1";
            this.button1.Primary = true;
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 7;
            this.button1.Text = "Ok";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.materialRaisedButton1_Click);
            // 
            // linkLabel2
            // 
            this.linkLabel2.AutoSize = true;
            this.linkLabel2.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.linkLabel2.LinkColor = System.Drawing.Color.Black;
            this.linkLabel2.Location = new System.Drawing.Point(9, 299);
            this.linkLabel2.Name = "linkLabel2";
            this.linkLabel2.Size = new System.Drawing.Size(61, 13);
            this.linkLabel2.TabIndex = 8;
            this.linkLabel2.TabStop = true;
            this.linkLabel2.Text = "Источники";
            this.linkLabel2.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel2_LinkClicked);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(12, 82);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(145, 168);
            this.pictureBox1.TabIndex = 3;
            this.pictureBox1.TabStop = false;
            // 
            // menuStrip1
            // 
            this.menuStrip1.BackColor = System.Drawing.Color.DarkSeaGreen;
            this.menuStrip1.BackgroundImage = global::Filosofiya.Properties.Resources.menustip;
            this.menuStrip1.Font = new System.Drawing.Font("Segoe UI", 9F);
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.настройкиToolStripMenuItem,
            this.темаToolStripMenuItem,
            this.настройкиШрифтаToolStripMenuItem,
            this.предпочтенияToolStripMenuItem,
            this.выходToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(690, 24);
            this.menuStrip1.TabIndex = 5;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // настройкиToolStripMenuItem
            // 
            this.настройкиToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.уведомленияToolStripMenuItem,
            this.типToolStripMenuItem,
            this.таймерToolStripMenuItem,
            this.автозагрузкаToolStripMenuItem,
            this.выводЦитатыToolStripMenuItem,
            this.записатьЦитатуToolStripMenuItem,
            this.обновлениеЦитатыToolStripMenuItem});
            this.настройкиToolStripMenuItem.ForeColor = System.Drawing.SystemColors.ButtonFace;
            this.настройкиToolStripMenuItem.Name = "настройкиToolStripMenuItem";
            this.настройкиToolStripMenuItem.Size = new System.Drawing.Size(79, 20);
            this.настройкиToolStripMenuItem.Text = "Настройки";
            // 
            // уведомленияToolStripMenuItem
            // 
            this.уведомленияToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripComboBox1});
            this.уведомленияToolStripMenuItem.Name = "уведомленияToolStripMenuItem";
            this.уведомленияToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.уведомленияToolStripMenuItem.Text = "Тип уведомлений";
            // 
            // toolStripComboBox1
            // 
            this.toolStripComboBox1.Items.AddRange(new object[] {
            "Уведомления_1",
            "Уведомления_2",
            "Отключить"});
            this.toolStripComboBox1.Name = "toolStripComboBox1";
            this.toolStripComboBox1.Size = new System.Drawing.Size(121, 23);
            this.toolStripComboBox1.Text = "Выбрать уведом.";
            // 
            // типToolStripMenuItem
            // 
            this.типToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripComboBox3});
            this.типToolStripMenuItem.Name = "типToolStripMenuItem";
            this.типToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.типToolStripMenuItem.Text = "Часы";
            // 
            // toolStripComboBox3
            // 
            this.toolStripComboBox3.IntegralHeight = false;
            this.toolStripComboBox3.Items.AddRange(new object[] {
            "Часы(ON)",
            "Часы(OFF)"});
            this.toolStripComboBox3.Name = "toolStripComboBox3";
            this.toolStripComboBox3.Size = new System.Drawing.Size(121, 23);
            this.toolStripComboBox3.Text = "Вкл,выкл.";
            this.toolStripComboBox3.TextChanged += new System.EventHandler(this.toolStripComboBox3_TextChanged);
            // 
            // таймерToolStripMenuItem
            // 
            this.таймерToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripComboBox4});
            this.таймерToolStripMenuItem.Name = "таймерToolStripMenuItem";
            this.таймерToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.таймерToolStripMenuItem.Text = "Таймер";
            this.таймерToolStripMenuItem.Click += new System.EventHandler(this.таймерToolStripMenuItem_Click);
            // 
            // toolStripComboBox4
            // 
            this.toolStripComboBox4.Items.AddRange(new object[] {
            "On",
            "Off"});
            this.toolStripComboBox4.Name = "toolStripComboBox4";
            this.toolStripComboBox4.Size = new System.Drawing.Size(121, 23);
            this.toolStripComboBox4.Text = "Вкл,выкл.";
            this.toolStripComboBox4.SelectedIndexChanged += new System.EventHandler(this.toolStripComboBox4_SelectedIndexChanged);
            // 
            // автозагрузкаToolStripMenuItem
            // 
            this.автозагрузкаToolStripMenuItem.Name = "автозагрузкаToolStripMenuItem";
            this.автозагрузкаToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.автозагрузкаToolStripMenuItem.Text = "Автозагрузка";
            this.автозагрузкаToolStripMenuItem.Click += new System.EventHandler(this.автозагрузкаToolStripMenuItem_Click);
            // 
            // выводЦитатыToolStripMenuItem
            // 
            this.выводЦитатыToolStripMenuItem.Name = "выводЦитатыToolStripMenuItem";
            this.выводЦитатыToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.выводЦитатыToolStripMenuItem.Text = "Вывод цитаты";
            this.выводЦитатыToolStripMenuItem.Click += new System.EventHandler(this.выводЦитатыToolStripMenuItem_Click);
            // 
            // записатьЦитатуToolStripMenuItem
            // 
            this.записатьЦитатуToolStripMenuItem.Name = "записатьЦитатуToolStripMenuItem";
            this.записатьЦитатуToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.записатьЦитатуToolStripMenuItem.Text = "Добавить цитату";
            this.записатьЦитатуToolStripMenuItem.Click += new System.EventHandler(this.записатьЦитатуToolStripMenuItem_Click);
            // 
            // обновлениеЦитатыToolStripMenuItem
            // 
            this.обновлениеЦитатыToolStripMenuItem.Name = "обновлениеЦитатыToolStripMenuItem";
            this.обновлениеЦитатыToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.обновлениеЦитатыToolStripMenuItem.Text = "Обновление цитат";
            this.обновлениеЦитатыToolStripMenuItem.Click += new System.EventHandler(this.обновлениеЦитатыToolStripMenuItem_Click);
            // 
            // темаToolStripMenuItem
            // 
            this.темаToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripComboBox2});
            this.темаToolStripMenuItem.ForeColor = System.Drawing.SystemColors.Control;
            this.темаToolStripMenuItem.Name = "темаToolStripMenuItem";
            this.темаToolStripMenuItem.Size = new System.Drawing.Size(93, 20);
            this.темаToolStripMenuItem.Text = "Оформление";
            // 
            // toolStripComboBox2
            // 
            this.toolStripComboBox2.Items.AddRange(new object[] {
            "Темная",
            "Светлая"});
            this.toolStripComboBox2.Name = "toolStripComboBox2";
            this.toolStripComboBox2.Size = new System.Drawing.Size(121, 23);
            this.toolStripComboBox2.Text = "Выбрать тему:";
            this.toolStripComboBox2.SelectedIndexChanged += new System.EventHandler(this.toolStripComboBox2_SelectedIndexChanged);
            this.toolStripComboBox2.Click += new System.EventHandler(this.toolStripComboBox2_Click);
            // 
            // настройкиШрифтаToolStripMenuItem
            // 
            this.настройкиШрифтаToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.настройкиШрифтаЧасовToolStripMenuItem,
            this.настройкиШрифтаЦитатыToolStripMenuItem});
            this.настройкиШрифтаToolStripMenuItem.ForeColor = System.Drawing.SystemColors.Control;
            this.настройкиШрифтаToolStripMenuItem.Name = "настройкиШрифтаToolStripMenuItem";
            this.настройкиШрифтаToolStripMenuItem.Size = new System.Drawing.Size(127, 20);
            this.настройкиШрифтаToolStripMenuItem.Text = "Настройки шрифта";
            // 
            // настройкиШрифтаЧасовToolStripMenuItem
            // 
            this.настройкиШрифтаЧасовToolStripMenuItem.Name = "настройкиШрифтаЧасовToolStripMenuItem";
            this.настройкиШрифтаЧасовToolStripMenuItem.Size = new System.Drawing.Size(224, 22);
            this.настройкиШрифтаЧасовToolStripMenuItem.Text = "Настройки шрифта часов";
            this.настройкиШрифтаЧасовToolStripMenuItem.Click += new System.EventHandler(this.настройкиШрифтаЧасовToolStripMenuItem_Click);
            // 
            // настройкиШрифтаЦитатыToolStripMenuItem
            // 
            this.настройкиШрифтаЦитатыToolStripMenuItem.Name = "настройкиШрифтаЦитатыToolStripMenuItem";
            this.настройкиШрифтаЦитатыToolStripMenuItem.Size = new System.Drawing.Size(224, 22);
            this.настройкиШрифтаЦитатыToolStripMenuItem.Text = "Настройки шрифта цитаты";
            this.настройкиШрифтаЦитатыToolStripMenuItem.Click += new System.EventHandler(this.настройкиШрифтаЦитатыToolStripMenuItem_Click);
            // 
            // предпочтенияToolStripMenuItem
            // 
            this.предпочтенияToolStripMenuItem.ForeColor = System.Drawing.SystemColors.Control;
            this.предпочтенияToolStripMenuItem.Name = "предпочтенияToolStripMenuItem";
            this.предпочтенияToolStripMenuItem.Size = new System.Drawing.Size(99, 20);
            this.предпочтенияToolStripMenuItem.Text = "Предпочтения";
            this.предпочтенияToolStripMenuItem.Click += new System.EventHandler(this.предпочтенияToolStripMenuItem_Click);
            // 
            // выходToolStripMenuItem
            // 
            this.выходToolStripMenuItem.ForeColor = System.Drawing.SystemColors.Control;
            this.выходToolStripMenuItem.Name = "выходToolStripMenuItem";
            this.выходToolStripMenuItem.Size = new System.Drawing.Size(53, 20);
            this.выходToolStripMenuItem.Text = "Выход";
            this.выходToolStripMenuItem.Click += new System.EventHandler(this.выходToolStripMenuItem_Click);
            // 
            // основное1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(690, 321);
            this.Controls.Add(this.linkLabel2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.linkLabel1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.menuStrip1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "основное1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = " ";
            this.Load += new System.EventHandler(this.основное1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.LinkLabel linkLabel1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem настройкиToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem темаToolStripMenuItem;
        private System.Windows.Forms.ToolStripComboBox toolStripComboBox2;
        private System.Windows.Forms.ToolStripMenuItem выходToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem типToolStripMenuItem;
        private System.Windows.Forms.ToolStripComboBox toolStripComboBox3;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Timer timer2;
        private System.Windows.Forms.ToolStripMenuItem таймерToolStripMenuItem;
        private System.Windows.Forms.ToolStripComboBox toolStripComboBox4;
        private System.Windows.Forms.ToolStripMenuItem записатьЦитатуToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem уведомленияToolStripMenuItem;
        private System.Windows.Forms.ToolStripComboBox toolStripComboBox1;
        private System.Windows.Forms.ToolStripMenuItem настройкиШрифтаToolStripMenuItem;
        private System.Windows.Forms.FontDialog fontDialog1;
        private System.Windows.Forms.ToolStripMenuItem настройкиШрифтаЧасовToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem настройкиШрифтаЦитатыToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem выводЦитатыToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem обновлениеЦитатыToolStripMenuItem;
        private MaterialSkin.Controls.MaterialRaisedButton button1;
        private System.Windows.Forms.ToolStripMenuItem предпочтенияToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem автозагрузкаToolStripMenuItem;
        private System.Windows.Forms.LinkLabel linkLabel2;
    }
}