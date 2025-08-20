#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // dirección 0x27, LCD 16x2
String input = "";

void setup() {
  lcd.init();
  lcd.backlight();
  lcd.print("Esperando voz...");
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Nivel de voz:");
      lcd.setCursor(0, 1);
      lcd.print(input);
      input = "";
    } else {
      input += c;
    }
  }
}
