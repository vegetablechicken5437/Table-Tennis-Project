varying vec4 vColor;
varying vec3 N;
varying vec3 v;    

uniform sampler2D THL_DiffuseMap;   //0 shoud be defined in shader program (by glUniform1i)
uniform sampler2D THL_NormalMap;    //1 shoud be defined in shader program (by glUniform1i)
varying mat3 TBNMatrix;

void main (void)  
{  
  vec3 L = normalize(gl_LightSource[0].position.xyz - v);   
  vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0)  
  
 
  //calculate Ambient Term:  
  //calculate Diffuse Term:  
  //calculate Specular Term:
  vec4 Iamb;
  vec4 Idiff;
  vec4 Ispec;

    if ( vColor.xyzw == vec4(0,0,0,0) ) // No TEXTURE
	{
			vec3 R = normalize(-reflect(L,N));  
			
			Iamb = gl_LightSource[0].ambient*gl_FrontMaterial.ambient;
			Idiff = gl_LightSource[0].diffuse * max( dot(N,L), 0.0 ) * gl_FrontMaterial.diffuse;
			Ispec = gl_LightSource[0].specular * pow( max( dot(R,E) , 0.0 ), gl_FrontMaterial.shininess) * gl_FrontMaterial.specular;
	}
   	else if ( vColor.xyzw == vec4(1,1,1,0) )  //Texture
	{
			vec3 norm = (TBNMatrix*(texture2D(THL_NormalMap, vec2(gl_TexCoord[1])).rgb * 2.0 - 1.0 )) ;
			vec3 R = normalize(-reflect(L,norm));  			
			
			Iamb = gl_LightSource[0].ambient* texture2D(THL_DiffuseMap, vec2(gl_TexCoord[0]));    
			Idiff = gl_LightSource[0].diffuse * max( dot( norm , L), 0.0 ) * texture2D(THL_DiffuseMap, vec2(gl_TexCoord[0]));
			Ispec = gl_LightSource[0].specular * pow( max(0.0,dot(R, E)), gl_FrontMaterial.shininess) * gl_FrontMaterial.specular;
	}
	else if ( vColor.xyzw == vec4(0.5,0.5,0.5,0) )  //FLAT Texture
	{
			Iamb = texture2D(THL_NormalMap, vec2(gl_TexCoord[0]));  // For flat texture
	}
	else // Vertex Color
	{
			vec3 R = normalize(-reflect(L,N));  
			
			Iamb = gl_LightSource[0].ambient *vColor;
			Idiff = gl_LightSource[0].diffuse * max( dot(N,L), 0.0 ) * vColor;
			Ispec = gl_LightSource[0].specular * pow( max( dot(R,E) , 0.0 ), gl_FrontMaterial.shininess) * gl_FrontMaterial.specular;
	}

   Iamb = clamp(Iamb, 0.0, 1.0);     
   Idiff = clamp(Idiff, 0.0, 1.0);     
   Ispec = clamp(Ispec, 0.0, 1.0); 

   // write Total Color:  
   gl_FragColor = Iamb + Idiff + Ispec;     
}         